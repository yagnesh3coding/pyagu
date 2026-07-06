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
        # x shape: [batch_size, audio_seq_len, 4]
        h = self.proj(x)
        return self.layernorm(h)

class HyperPersonaNetwork(nn.Module):
    """
    Generates FiLM (Feature-wise Linear Modulation) scale & shift parameters 
    for each layer Norm in the transformer block from the Persona ID.
    This dynamically alters the weights/routing depending on active character.
    """
    def __init__(self, num_personas, num_layers, d_model):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        # Output is 2 (scale & shift) * num_layers * d_model
        self.hyper = nn.Sequential(
            nn.Embedding(num_personas, 64),
            nn.ReLU(),
            nn.Linear(64, num_layers * 2 * d_model)
        )

    def forward(self, persona_id):
        # persona_id: [batch_size]
        params = self.hyper(persona_id) # [batch, num_layers * 2 * d_model]
        params = params.view(-1, self.num_layers, 2, self.d_model)
        return params # [batch, num_layers, 2, d_model]

class TokenGatedMemory(nn.Module):
    """
    Differentiable Read-Write Working Memory.
    Reads from and writes to memory slots at each sequence token step.
    """
    def __init__(self, d_model, num_slots=4):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        
        # Read components
        self.mha = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.read_norm = nn.LayerNorm(d_model)
        
        # Write components
        self.q_proj = nn.Linear(d_model, d_model)
        self.erase_proj = nn.Linear(d_model, d_model)
        self.add_proj = nn.Linear(d_model, d_model)
        
    def forward(self, h, memory_slots):
        """
        h: token representations [batch, seq_len, d_model]
        memory_slots: memory bank [batch, num_slots, d_model]
        """
        batch_size, seq_len, _ = h.size()
        
        # 1. Read from memory bank via Cross-Attention
        # Query: token hidden states, Key/Value: memory slots
        read_val, _ = self.mha(h, memory_slots, memory_slots)
        h = self.read_norm(h + read_val)
        
        # 2. Token-level dynamic write-gate updates (sequential across seq_len)
        updated_slots = memory_slots.clone()
        
        for t in range(seq_len):
            xt = h[:, t, :] # [batch, d_model]
            
            # Compute write key and erase/add vectors
            q_t = self.q_proj(xt) # [batch, d_model]
            erase_t = torch.sigmoid(self.erase_proj(xt)) # [batch, d_model]
            add_t = torch.tanh(self.add_proj(xt)) # [batch, d_model]
            
            # Addressing scores: similarity between write key and memory slots
            # Shape: [batch, num_slots]
            scores = F.softmax(torch.bmm(updated_slots, q_t.unsqueeze(2)).squeeze(2) / (self.d_model ** 0.5), dim=-1)
            
            # Apply Gated Erase and Add updates
            # M_i = M_i * (1 - scores_i * erase_t) + scores_i * add_t
            erase_gate = scores.unsqueeze(2) * erase_t.unsqueeze(1) # [batch, num_slots, d_model]
            add_gate = scores.unsqueeze(2) * add_t.unsqueeze(1)     # [batch, num_slots, d_model]
            
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
        """
        token_ids: [batch, seq_len]
        emotion_state: [batch, num_emotions]
        memory_slots: [batch, num_slots, d_model]
        persona_id: [batch]
        image: [batch, 3, 32, 32] or None
        audio: [batch, audio_seq_len, 4] or None
        """
        batch_size, seq_len = token_ids.size()
        
        # 1. Embed and concatenate multi-modal tokens
        tokens = self.token_embed(token_ids) + self.pos_embed[:, :seq_len, :]
        
        inputs_list = []
        if image is not None:
            inputs_list.append(self.vision_encoder(image)) # Visual tokens
        if audio is not None:
            inputs_list.append(self.audio_encoder(audio)) # Audio tokens
            
        inputs_list.append(tokens)
        h = torch.cat(inputs_list, dim=1) # Unified sequence embedding
        
        # 2. Get Persona Layer Modulation Factors (FiLM)
        # film_params shape: [batch, num_layers, 2, d_model]
        film_params = self.hyper_persona(persona_id)
        
        # 3. Decode sequence, applying layer modulation
        for idx, layer in enumerate(self.layers):
            # Normal forward pass through transformer block
            h = layer(tgt=h, memory=memory_slots)
            
            # Apply FiLM Layer modulation (scale & shift)
            gamma = film_params[:, idx, 0, :].unsqueeze(1) # [batch, 1, d_model]
            beta = film_params[:, idx, 1, :].unsqueeze(1)  # [batch, 1, d_model]
            h = gamma * h + beta
            
        # Extract text sequence representation
        text_h = h[:, -seq_len:, :]
        
        # 4. Update memory bank token-by-token
        text_h, next_memory_slots = self.memory_controller(text_h, memory_slots)
        
        # 5. Emotion Dynamics
        last_h = text_h[:, -1, :]
        emotion_delta = self.emotion_net(last_h)
        next_emotion_state = 0.8 * emotion_state + 0.2 * emotion_delta
        next_emotion_state = torch.clamp(next_emotion_state, -1.0, 1.0)
        
        # 6. Apply emotional bias to predicted token logits
        logits = self.output_head(text_h)
        ebias = self.emotion_bias_map(next_emotion_state).unsqueeze(1)
        logits = logits + ebias
        
        return logits, next_emotion_state, next_memory_slots
