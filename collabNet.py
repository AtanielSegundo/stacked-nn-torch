import numpy as np
from typing import Optional, Tuple, List
from enum import Enum
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

'''
[Ataniel - 10/11/2025]

******************CONSIDERAÇÕES******************

1) Variantes de Metodologia:

1.1) A metodologia tem duas variacoes
	a) Fator k entre camada oculta O[i-1] e O[i]
	   quanto ao seu treinamento 
	   k_changes = [BOOL| True,False]
	b) Extra branch
	   que recebe o foward do vetor
	   de features e propaga para a nova
	   camada oculta da rede
	   Ext[InputDim,NextHiddenDim] = None

1.2) Mutação da função de ativação
	1.2.1) Normal: 
		   Funcao de Ativacao inalterada(linear)
	1.2.2) ChangeOut:
		   Na camada de saida anterior
		   Funcao de ativacao interpola de
		   linear para não linear
		   (lerp(X,linear,GeLu,eta))
	1.2.3) ChangeHidden:
		   Na camada oculta nova
		   Funcao de ativacao interpola de
		   linear para não linear
		   (lerp(X,linear,GeLu,eta))
	
	Nota-se que em ambos os casos de mutação

	a camada (saida/oculta) é linear e se torna
	não linear, implica que iniciam linear

	Na IMPLEMENTAÇÃO serão necessarios:
	
	eta: 
		Coeficiente de interpolação 
		Numero Real entre (0,1)
	TargetFn:
		Função não linear alvo
		(Tanh,ReLu,GeLu,Sigmoid)

	Precisa-se de um dois parametros:
		MutationMode: str = 
					 (None [padrão] | "Out" | "Hidden")
		targetFn: torch.nn.function = None [padrão]

2) Resumo da metodologia:

	A metodologia propõe-se a criar uma rede
	baseada em Stack AutoEncoders que adiciona
	novas camadas a mesma de modo que o erro da
	rede diminua

2.1) Exemplo de fluxo tipico da rede

[1] - Primeira camada adicionada na rede

ELEMENTOS:
	N	 := Tamanho do vetor de entrada
	_H1  := Tamanho do vetor da camada oculta
	_O1  := Tamanho do vetor da camada de saida

	X    := Vetor de Features [N,1]
			(x0,x1,...,xn)
	H{1} := Vetor da Camada oculta [_H1,1]
			(h0,h1,...,hn)

	W{1} := Pesos [_H1,N]
	B{1} := Vieses [_H1,1]

	O{1} := Vetor de saida [_O1,1]
	W{2} := Pesos [_O1,_H1]
	B{2} := Vieses [_O1,1]

OBS: TODOS OS PESOS SÃO INICIALIZADOS ALEATORIAMENTE

FLUXO DE INFORMAÇÃO:
f := Qualquer função de ativação (linear...Softmax)

H{1} = f(W{1}X + B{1})
O{1} = f(W{2}H{1} + B{2})

X --> H{1} --> O{1}

PYTORCH:
Em sintese para reproduzir 
esse fluxo inicial sera necessario

L{1} = nn.Linear(N,_H1)
L{2} = nn.Linear(_H1,_O1)

H{1} = f{1}(L{1}(X))
O{1} = f{2}(L{2}(H{1}))

[2] - Segunda camada adicionada na rede

Supõe-se que a rede foi treinada como um todo
ou seja só a primeira camada e atingiu um ponto
de estagnação (Sem diminuição na Loss Function)

Elementos:
	K   := Fator k para os pesos entre 
		   camadas ocultas
	_E1 := Tamanho do vetor da camada extra
	_N2 := Tamanho do vetor de entrada para
		   a segunda rede
	_N2 = _H1
	_H2  := Tamanho do vetor da nova
			camada oculta
	_O2  := Tamanho do vetor da nova camada de saida
	
	eta := interpolation coef
	targetFn := Funcao alvo
	MutationMode := Modo de mutação

	X    := Vetor de Features [N,1]
			(x0,x1,...,xn)

	H{2} := Vetor da Camada oculta [_H2,1]
			(h0,h1,...,hn)
	O{2} := Vetor de saida [_O2,1]
	E{1} := Vetor de ponderacao extra [_E1,1]

	W{2+1} := Pesos [_H2,_N2]
	B{2+1} := Vieses [_H2,1]

	W{extra+1} := Pesos [_E1,N]
	B{extra+1} := Vieses [_E1,1]

	W{extra+2} := Pesos [_H2,_E1]
	B{extra+2} := Vieses [_H2,1]

	W{2+2} := Pesos [_O2,_H2]
	B{2+2} := Vieses [_O2,1]

	W{2+3} := Pesos [_O2,_O1]
	B{2+3} := Vieses [_O2,1]

	OBS: Sera usado o MutationMode = "Hidden",
		 pois o "Out" é apenas um caso particular,
		 onde a targetFn de já esta sendo usada
		 diretamente nos pesos W{2+2} e é necessario
		 interpolar a targetFn em 

FLUXO DE INFORMAÇÃO:

f := Qualquer função de ativação (linear...Softmax)

E{1} = f(W{extra+1}X + B{extra+1})
H{2} = f(W{2+1}H{1} + B{2+1} +
		 W{extra+2}E{1} + B{extra+2}
		)
Z = W{2+2}H{2} + B{2+2} +
	   W{2+3}O{1} + B{2+3}		
O{2} = Interpolate(Z,TargetFn,eta)

Caso MutationMode = "Out":
	Z = W{2+3}O{1} + B{2+3}
	Z = Interpolate(Z,TargetFn,eta)
	O{2} = Z + f(W{2+2}H{2} + B{2+2})

OBS:	PARA O FUNCIONAMENTO ADEQUADO
		OU SEJA QUE A ADIÇÃO DE UMA CAMADA
		NÃO GERE AUMENTA NA FUNCAO DE PERCA
		É NECESSARIO (K = 1.0):

	W{2+2} = {0}, B{2+2} = {0}, B{2+3} = {0}
	W{2+3} = K * EYE()[_O2,_O1]

OBS: OS DEMAIS PESOS PODEM SER INICIADOS
	 ALEATORIAMENTE
	 '*' = Congelado

X --> H*{1} --> O*{1} -->|
|	  |----> H{2} ------>O{2} 
|--> E{1} -->|

CONGELAMENTO DE CAMADAS:

O ATO DE ADICIONAR UMA NOVA CAMADA NA REDE
IMPLICA EM CONGELAR A CAMADA ANTERIOR 

Logo:

L{1} e L{2} serao congelados ou seja
require_grad = false

[3] - Terceira camada em diante adicionada na rede

A logica permanece a mesma da insercao da segunda
camada, entratando a um detalhe fundamental

H{3} deve receber em sua entrada a concatenacao
	 de O{1} e H{2}, pois ambas comecam a atuar
	 como se fossem apenas mais uma camada oculta

Generalizando 

H{N} recebe concatenacao de H{N-1} com O{N-2}

O resto segue a mesma logica anterior para
	- E{N-1} : Extra layer
	- Congelamento de pesos
	- logica de inicializacao para que nao 
	  haja aumento na loss function em relacao
	  a camada anterior
'''

def lerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    return (1.0 - t) * a + t * b

class MutationMode(Enum):
    Normal = 0
    Out    = 1
    Hidden = 2
    
# CLASS LIKE AN TAGGED STRUCTURE WITH AN GENERIC TYPE ASSOCIATED
from typing import _T
class LayersConfig:
    def __init__(self,hidden:_T,out:_T,extra:_T):
        self.hidden = hidden
        self.out    = out
        self.extra  = extra

class StackedLayer(nn.Module):
    """
    Representa um "bloco empilhado" que contém:
     - linear_hidden: entrada -> hidden
     - optional extra branch: original_input -> E -> contribui para hidden
     - linear_out: hidden -> out
     - optional skip/identity mapping from prev_out -> out (W{skip})
    Suporta MutationMode: None | "Hidden" | "Out"
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        original_input_dim: Optional[int] = None,
        extra_dim: Optional[int] = None,
        prev_out_dim: Optional[int] = None,
        k_skip: float = 1.0,
        mutation_mode: Optional[MutationMode] = None,
        target_fn: Optional[nn.Module] = None,
        eta: float = 0.0,
        eta_increment: float = 0.001,
        hidden_activation: Optional[nn.Module] = None,
        extra_activation: Optional[nn.Module] = None,
        out_activation: Optional[nn.Module] = None,
        is_first: bool = False,
        is_k_trainable=True,
        device: Optional[torch.device] = None,
        layer_use_bias:LayersConfig = None
    ):
        super().__init__()
        self.device = device or torch.device("cpu")

        self.input_dim    = input_dim
        self.hidden_dim   = hidden_dim
        self.out_dim      = out_dim
        self.prev_out_dim = prev_out_dim
        self.is_first     = is_first
        self.is_freezed   = False

        self.layer_use_bias    = layer_use_bias or LayersConfig(True,True,True) 
        self.hidden_activation = hidden_activation or nn.Identity()
        self.out_activation    = out_activation or nn.Identity()
        self.extra_activation  = extra_activation or nn.Identity()

        self.linear_hidden = nn.Linear(input_dim, hidden_dim, 
                                       bias=self.layer_use_bias.hidden)
        self.linear_out    = nn.Linear(hidden_dim, out_dim, 
                                       bias=self.layer_use_bias.out)

        if extra_dim is not None and original_input_dim is not None: 
            self.use_extra       = True
            self.extra_in        = nn.Linear(original_input_dim, extra_dim, 
                                             bias=self.layer_use_bias.extra)
            self.extra_to_hidden = nn.Linear(extra_dim, hidden_dim, 
                                             bias=self.layer_use_bias.extra)
        else: 
            self.use_extra       = False
            self.extra_in        = None
            self.extra_to_hidden = None

        if prev_out_dim is not None and not is_first: 
            self.use_skip    = True
            self.linear_skip = nn.Linear(prev_out_dim, out_dim, 
                                         bias=self.layer_use_bias.out)
        else: 
            self.use_skip    = False
            self.linear_skip = None

        self.mutation_mode  = mutation_mode
        self.target_fn      = target_fn
        self.eta            = float(eta)
        self.eta_increment  = float(eta_increment)
        self.eta_multiplier = 1.0

        if not is_first:
            # ensure new contribution from hidden -> out starts as zero
            with torch.no_grad():
                self.linear_out.weight.zero_()
                if self.linear_out.bias is not None:
                    self.linear_out.bias.zero_()
            
			# set skip to k * identity (or an identity-embedding)
            if self.use_skip:
                with torch.no_grad():
                    self.linear_skip.weight.zero_()
                    if self.linear_skip.bias is not None:
                        self.linear_skip.bias.zero_()
                    m = min(self.out_dim, self.prev_out_dim)
                    for i in range(m):
                        self.linear_skip.weight[i, i] = k_skip
                    if not is_k_trainable:
                        self.linear_skip.weight.requires_grad = False
                        self.linear_skip.bias.requires_grad   = False

        self.to(self.device)

    def step_eta(self):
        """Incrementa eta gradualmente até atingir 1.0"""
        if self.eta > 1.0: return
        eta = min(1.0, self.eta + self.eta_increment * self.eta_multiplier)
        is_out = self.mutation_mode is MutationMode.Out
        is_hidden = (self.mutation_mode is MutationMode.Hidden) and self.is_freezed
        self.eta = eta if is_out or is_hidden else self.eta
    
    def accelerate_eta(self, factor: float = 2.0):
        """Aumenta a velocidade de incremento do eta (chamado quando nova camada é adicionada)"""
        self.eta_multiplier *= factor

    def forward(
        self,
        hidden_input: torch.Tensor,
        prev_o: Optional[torch.Tensor] = None,
        original_x: Optional[torch.Tensor] = None,
    ):
        """
        hidden_input: tensor fed to linear_hidden (shape [batch, input_dim])
        prev_o: previous layer output O_{n-1} (shape [batch, prev_out_dim]) or None
        original_x: full original input X if extra branch is used
        Returns: (H, O)
        """
        
        h_pre = self.linear_hidden(hidden_input)

        if self.use_extra and (original_x is not None):
            e = self.extra_in(original_x)
            e = self.extra_activation(e)
            e = self.extra_to_hidden(e)
            h_pre = h_pre + e

        h_activated = self.hidden_activation(h_pre)
        h_out = self.linear_out(h_activated)

        if self.use_skip and (prev_o is not None):
            skip_contribution = self.linear_skip(prev_o)
        else:
            skip_contribution = 0

        z = h_out + skip_contribution

        if self.target_fn and self.mutation_mode is not None:
            if self.mutation_mode is MutationMode.Out:
                target_out = self.target_fn(z)
                o = lerp(z, target_out, self.eta)
            elif self.mutation_mode is MutationMode.Hidden:
                if self.is_freezed:
                    target_out = self.target_fn(z)
                    o = lerp(z, target_out, self.eta)
                else:
                    o = z
        else:
            o = z

        return h_activated, o
    
    def freeze(self):
        self.is_freezed = True
        for p in self.parameters():
            p.requires_grad = False


class SAECollabNet(nn.Module):
    """
    Rede que armazena camadas empilhadas (StackedLayer).
    """
    def __init__(
        self,
        input_dim: int,
        first_hidden: int,
        first_out: int,
        *,
        device: Optional[torch.device] = None,
        hidden_activation: Optional[nn.Module] = None,
        out_activation: Optional[nn.Module] = None,
        accelerate_etas: bool = False
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.input_dim = input_dim

        self.accelerate_etas = accelerate_etas
        self.layers = nn.ModuleList()
        self.hidden_dims: List[int] = []
        self.out_dims: List[int] = []

        # create first layer
        first = StackedLayer(
            input_dim=input_dim,
            hidden_dim=first_hidden,
            out_dim=first_out,
            original_input_dim=None,
            extra_dim=None,
            prev_out_dim=None,
            is_first=True,
            device=self.device,
            hidden_activation=hidden_activation,
            out_activation=out_activation,
            mutation_mode=None,
            target_fn=None,
            eta=0.0,
        )
        self.layers.append(first)
        self.hidden_dims.append(first_hidden)
        self.out_dims.append(first_out)
        self.to(self.device)

    def add_layer(
        self,
        hidden_dim: int,
        out_dim: int,
        *,
        extra_dim: Optional[int] = None,
        k: float = 1.0,
        mutation_mode: Optional[str] = None,
        target_fn: Optional[nn.Module] = None,
        eta: float = 0.0,
        eta_increment: float = 0.001,
        hidden_activation: Optional[nn.Module] = None,
        out_activation: Optional[nn.Module] = None,
        extra_activation: Optional[nn.Module] = None,
        accelerate_factor: float = 2.0,
        is_k_trainable=True,
    ):
        """
        Adiciona nova camada segundo a metodologia.
        Acelera o incremento de eta das camadas anteriores.
        """

        if self.accelerate_etas:
            for layer in self.layers:
                if layer.mutation_mode is not None:
                    layer.accelerate_eta(accelerate_factor)
        
        if len(self.layers) > 0:
            last = self.layers[-1]
            last.freeze()
            
        if len(self.layers) == 1:
            input_dim = self.hidden_dims[-1]
        else:
            # third or more: concatenation of H_{n-1} and O_{n-2}
            prev_hidden = self.hidden_dims[-1]
            prevprev_out = self.out_dims[-2]
            input_dim = prev_hidden + prevprev_out

        prev_out_dim = self.out_dims[-1] if len(self.out_dims) > 0 else None

        layer = StackedLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            original_input_dim=self.input_dim if extra_dim is not None else None,
            extra_dim=extra_dim,
            prev_out_dim=prev_out_dim,
            k_skip=k,
            mutation_mode=mutation_mode,
            target_fn=target_fn,
            eta=eta,
            eta_increment=eta_increment,
            hidden_activation=hidden_activation,
            out_activation=out_activation,
            extra_activation=extra_activation,
            is_first=False,
            is_k_trainable=is_k_trainable,
            device=self.device,
        )

        self.layers.append(layer)
        self.hidden_dims.append(hidden_dim)
        self.out_dims.append(out_dim)
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        """
        Executa a rede empilhada.
        x: [batch, input_dim]
        Retorna: o_final, list_of_hiddens, list_of_outputs
        """
        hs: List[torch.Tensor] = []
        os: List[torch.Tensor] = []

        # first layer: input is original x
        h, o = self.layers[0](x, prev_o=None, original_x=None)
        hs.append(h)
        os.append(o)

        for idx in range(1, len(self.layers)):
            layer = self.layers[idx]

            if idx == 1:
                hidden_input = hs[-1]
            else:
                hidden_input = torch.cat([hs[-1], os[-2]], dim=1)

            prev_o = os[-1]  # O_{n-1} for skip connection
            original_x = x if layer.use_extra else None

            h, o = layer(hidden_input, prev_o=prev_o, original_x=original_x)
            hs.append(h)
            os.append(o)

        return os[-1], os, hs

    def freeze_all_but_last(self):
        for layer in self.layers[:-1]:
            for p in layer.parameters():
                p.requires_grad = False

    def unfreeze_last(self):
        if self.layers:
            for p in self.layers[-1].parameters():
                p.requires_grad = True
    
    def step_all_etas(self):
        for layer in self.layers:
            if layer.mutation_mode is not None:
                layer.step_eta()
    
    def get_eta_status(self):
        status = []
        for idx, layer in enumerate(self.layers):
            if layer.mutation_mode is not None:
                status.append({
                    'layer': idx,
                    'eta': layer.eta,
                    'multiplier': layer.eta_multiplier,
                    'mode': layer.mutation_mode
                })
        return status
    
    def debug_eta_status(self):
        status = self.get_eta_status()
        for l_status in status:
            print()
            print("Layer: ",l_status["layer"])
            print("eta: ",l_status["eta"])
            print("multiplier: ",l_status["multiplier"])
            print("mutation: ",l_status["mode"])
            print()


def demo_identity_preservation_collabnet():
    """
    Testa se ao adicionar uma nova camada com k=1.0 e linear_out zerado,
    a saída da rede permanece idêntica.
    """
    print("Demo: identity preservation test (CollabNet)")

    # dataset simples
    N = 200
    in_dim = 4
    out_dim = 2
    X = np.random.randn(N, in_dim).astype(np.float32)
    trueW = np.random.randn(in_dim, out_dim).astype(np.float32)
    Y = X.dot(trueW)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cria rede base
    net = SAECollabNet(
        input_dim=in_dim,
        first_hidden=8,
        first_out=out_dim,
        device=device,
        hidden_activation=nn.Identity(),
        out_activation=nn.Identity(),
    )

    # treino rápido da primeira camada
    def quick_train_first():
        opt = optim.SGD([p for p in net.parameters() if p.requires_grad], lr=1e-2)
        loss_fn = nn.MSELoss()
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        yb = torch.tensor(Y, dtype=torch.float32, device=device)
        for _ in range(100):
            pred, _, _ = net(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item()

    baseline_loss = quick_train_first()
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        baseline_out, _, _ = net(xb)
    baseline_out = baseline_out.cpu().numpy()

    # adiciona nova camada com k=1.0 (identidade)
    net.add_layer(
        hidden_dim=6,
        out_dim=out_dim,
        extra_dim=None,
        k=1.0,
        mutation_mode=None,
        target_fn=None,
        eta=0.0,
    )

    # sem treino, saída deve ser igual
    with torch.no_grad():
        new_out, _, _ = net(xb)
    new_out = new_out.cpu().numpy()

    max_diff = float(np.max(np.abs(new_out - baseline_out)))
    print(f"Max abs difference before training new layer: {max_diff:.10f}")

    if max_diff < 1e-5:
        print("✓ Identity preservation test PASSED.")
    else:
        print(f"✗ Identity preservation test FAILED. Difference: {max_diff}")


def demo_train_with_new_layer():
    """
    Testa se adicionar uma nova camada ao CollabNet reduz o erro MSE.
    """
    print("\nDemo: train with new layer improves MSE on simple regression")

    # Dataset simples y = 2x0 - x1 + 0.5x2 + ruído
    N = 500
    in_dim = 3
    out_dim = 1
    X = np.random.randn(N, in_dim).astype(np.float32)
    W_true = np.array([[2.0], [-1.0], [0.5]], dtype=np.float32)
    Y = (X.dot(W_true) + 0.1 * np.random.randn(N, 1)).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Inicializa rede com a primeira camada
    net = SAECollabNet(
        input_dim=in_dim,
        first_hidden=6,
        first_out=out_dim,
        device=device,
        hidden_activation=nn.ReLU(),
        out_activation=nn.Identity(),
    )

    loss_fn = nn.MSELoss()

    # Treina a primeira camada
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    yb = torch.tensor(Y, dtype=torch.float32, device=device)
    opt = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-2)
    losses_first = []

    for ep in range(100):
        pred, _, _ = net(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses_first.append(loss.item())

    with torch.no_grad():
        mse_before = float(loss_fn(net(xb)[0], yb).item())
    print(f"MSE after first layer train: {mse_before:.6f}")

    # 2) Adiciona nova camada
    net.add_layer(
        hidden_dim=8,
        out_dim=out_dim,
        extra_dim=None,
        k=1.0,
        mutation_mode=None,
        target_fn=None,
        eta=0.0,
    )

    with torch.no_grad():
        mse_preserve = float(loss_fn(net(xb)[0], yb).item())
    print(f"MSE right after adding layer: {mse_preserve:.6f}")

    # 3) Treina a segunda camada
    params_to_train = [p for p in net.parameters() if p.requires_grad]
    opt2 = optim.SGD(params_to_train, lr=1e-2)
    losses_second = []

    for ep in range(300):
        pred, _, _ = net(xb)
        loss = loss_fn(pred, yb)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        losses_second.append(loss.item())

    with torch.no_grad():
        mse_after = float(loss_fn(net(xb)[0], yb).item())
    print(f"MSE after training second layer: {mse_after:.6f}")

    # 4) Plota evolução das perdas
    plt.figure(figsize=(10,5))
    plt.plot(losses_first, label="Layer 1 training loss")
    plt.plot(range(len(losses_first),
                   len(losses_first) + len(losses_second)),
             losses_second, label="Layer 2 training loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("CollabNet Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("CUDA AVAILABLE: ", torch.cuda.is_available())
    demo_identity_preservation_collabnet()
    demo_train_with_new_layer()