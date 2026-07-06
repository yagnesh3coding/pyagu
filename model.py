import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionProjection(nn.Module):
    """
    Downsamples a 3x32x32 image into a sequence of 4 visual tokens.
    """
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.proj = nn.Linear(64, d_model)

    def forward(self, x):
        features = self.conv(x)
        features = self.pool(features)
        features = features.flatten(2).transpose(1, 2)
        return self.proj(features)

class AudioProjection(nn.Module):
    """
    Projects raw continuous sound features (e.g. 4 features: pitch, volume, spectral centroid, zero-crossing rate)
    into the shared d_model embedding space as audio tokens.
    """
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(4, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.proj(x)
        return self.layernorm(h)

class GrammarBlock(nn.Module):
    """
    Learns character transition rules and local spelling/grammatical patterns.
    Uses 1D Convolutions with kernel size 3 and 5 to capture n-gram structures.
    """
    def __init__(self, d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        c1 = F.relu(self.conv1(x_t))
        c2 = F.relu(self.conv2(x_t))
        out = torch.cat([c1, c2], dim=1).transpose(1, 2)
        return self.norm(x + self.proj(out))

class LexiconConceptSpace(nn.Module):
    """
    Maps syntactic representations into semantic word/concept embeddings.
    Also aligns these concepts with visual and audio modalities to ground their meanings.
    """
    def __init__(self, d_model):
        super().__init__()
        self.concept_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.vision_ground = nn.Linear(d_model, d_model)
        self.audio_ground = nn.Linear(d_model, d_model)

    def forward(self, concept_h, visual_h=None, audio_h=None):
        concepts = self.concept_proj(concept_h)
        grounding_loss = torch.tensor(0.0, device=concept_h.device)
        
        if visual_h is not None and visual_h.size(1) > 0:
            v_rep = self.vision_ground(visual_h.mean(dim=1))
            t_rep = concepts.mean(dim=1)
            cos_sim = F.cosine_similarity(t_rep, v_rep, dim=-1)
            grounding_loss = grounding_loss + (1.0 - cos_sim.mean())
            
        if audio_h is not None and audio_h.size(1) > 0:
            a_rep = self.audio_ground(audio_h.mean(dim=1))
            t_rep = concepts.mean(dim=1)
            cos_sim = F.cosine_similarity(t_rep, a_rep, dim=-1)
            grounding_loss = grounding_loss + (1.0 - cos_sim.mean())
            
        return concepts, grounding_loss

class HyperPersonaNetwork(nn.Module):
    """
    Generates FiLM (Feature-wise Linear Modulation) scale & shift parameters 
    for each layer Norm in the transformer block from the Persona ID.
    """
    def __init__(self, num_personas, num_layers, d_model):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.hyper = nn.Sequential(
            nn.Embedding(num_personas, 64),
            nn.ReLU(),
            nn.Linear(64, num_layers * 2 * d_model)
        )

    def forward(self, persona_id):
        params = self.hyper(persona_id)
        params = params.view(-1, self.num_layers, 2, self.d_model)
        return params

class TokenGatedMemory(nn.Module):
    """
    Differentiable Read-Write Working Memory.
    Reads from and writes to memory slots at each sequence token step.
    """
    def __init__(self, d_model, num_slots=4):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.mha = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.read_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.erase_proj = nn.Linear(d_model, d_model)
        self.add_proj = nn.Linear(d_model, d_model)
        
    def forward(self, h, memory_slots):
        batch_size, seq_len, _ = h.size()
        read_val, _ = self.mha(h, memory_slots, memory_slots)
        h = self.read_norm(h + read_val)
        updated_slots = memory_slots.clone()
        
        for t in range(seq_len):
            xt = h[:, t, :]
            q_t = self.q_proj(xt)
            erase_t = torch.sigmoid(self.erase_proj(xt))
            add_t = torch.tanh(self.add_proj(xt))
            
            scores = F.softmax(torch.bmm(updated_slots, q_t.unsqueeze(2)).squeeze(2) / (self.d_model ** 0.5), dim=-1)
            erase_gate = scores.unsqueeze(2) * erase_t.unsqueeze(1)
            add_gate = scores.unsqueeze(2) * add_t.unsqueeze(1)
            updated_slots = updated_slots * (1.0 - erase_gate) + add_gate
            
        return h, updated_slots

class PyAguModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3, num_slots=4, num_emotions=4, num_personas=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Multi-modal Input Projectors
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, d_model))
        self.vision_encoder = VisionProjection(d_model)
        self.audio_encoder = AudioProjection(d_model)
        
        # Developmental Hierarchy Layers
        self.grammar_block = GrammarBlock(d_model)
        self.lexicon_space = LexiconConceptSpace(d_model)
        
        # Dynamic Persona / HyperPersona Modulation
        self.hyper_persona = HyperPersonaNetwork(num_personas, num_layers, d_model)
        
        # Transformer Decoder layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Token-Level Gated Memory
        self.memory_controller = TokenGatedMemory(d_model, num_slots)
        
        # Emotion Network
        self.emotion_net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_emotions),
            nn.Tanh()
        )
        
        # Output Heads
        self.output_head = nn.Linear(d_model, vocab_size)
        self.emotion_bias_map = nn.Linear(num_emotions, vocab_size, bias=False)

    def forward(self, token_ids, emotion_state, memory_slots, persona_id, image=None, audio=None):
        batch_size, seq_len = token_ids.size()
        
        # 1. Developmental Stage 1: Syntactic/Grammar Embeddings
        tokens = self.token_embed(token_ids) + self.pos_embed[:, :seq_len, :]
        syntax_h = self.grammar_block(tokens)
        
        # 2. Extract visual and audio representations if present
        visual_tokens = self.vision_encoder(image) if image is not None else None
        audio_tokens = self.audio_encoder(audio) if audio is not None else None
        
        # 3. Developmental Stage 2 & 3: Word/Concept Lexicon and Cross-modal Grounding
        concepts, grounding_loss = self.lexicon_space(syntax_h, visual_tokens, audio_tokens)
        
        # 4. Integrate unified sequence embeddings
        inputs_list = []
        if visual_tokens is not None:
            inputs_list.append(visual_tokens)
        if audio_tokens is not None:
            inputs_list.append(audio_tokens)
            
        inputs_list.append(concepts)
        h = torch.cat(inputs_list, dim=1)
        
        # 5. Get Persona Layer Modulation Factors (FiLM)
        film_params = self.hyper_persona(persona_id)
        
        # 6. Decode sequence, applying layer modulation
        for idx, layer in enumerate(self.layers):
            h = layer(tgt=h, memory=memory_slots)
            gamma = film_params[:, idx, 0, :].unsqueeze(1)
            beta = film_params[:, idx, 1, :].unsqueeze(1)
            h = gamma * h + beta
            
        # Extract text sequence representation
        text_h = h[:, -seq_len:, :]
        
        # 7. Update memory bank token-by-token
        text_h, next_memory_slots = self.memory_controller(text_h, memory_slots)
        
        # 8. Emotion Dynamics
        last_h = text_h[:, -1, :]
        emotion_delta = self.emotion_net(last_h)
        next_emotion_state = 0.8 * emotion_state + 0.2 * emotion_delta
        next_emotion_state = torch.clamp(next_emotion_state, -1.0, 1.0)
        
        # 9. Apply emotional bias to predicted token logits
        logits = self.output_head(text_h)
        ebias = self.emotion_bias_map(next_emotion_state).unsqueeze(1)
        logits = logits + ebias
        
        return logits, next_emotion_state, next_memory_slots, grounding_loss
