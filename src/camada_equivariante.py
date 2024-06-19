import torch as t
from torch import nn
import numpy as np
from dataset_nbody import NBodyDataset
import torch.optim as optim
import json

class CamadaEquivariante(nn.Module):
    """ Uma camada equivariante utilizando convolução.
    """

    def __init__(self, entrada_nf: int, saida_nf: int, oculta_nf: int, ij: int):
        """ Inicializa uma camada equivariante.

        :param entrada_nf: Número de features de entrada de h
        :type entrada_nf: int
        :param saida_nf: Número de features de saída de h
        :type saida_nf: int
        :param oculta_nf: Número de neurônios na camada oculta
        :type oculta_nf: int
        :param ij: Dimensão do vetor de atributos das arestas
        :type ij: int
        """        
        super(CamadaEquivariante, self).__init__()
        self.entrada_nf = entrada_nf
        self.saida_nf = saida_nf
        self.oculta_nf = oculta_nf
        self.ij = ij

        # Se travar, xavier uniform
        self.phi_x = nn.Sequential(nn.Linear(oculta_nf, oculta_nf),
                                    nn.Tanh(),
                                    nn.Linear(oculta_nf, 1),
                                    nn.Tanh())
        
        # phi_e implementar
        self.phi_e = nn.Sequential(nn.Linear(entrada_nf * 2 + 1 + ij, oculta_nf),
                                    nn.Tanh(),
                                    nn.Linear(oculta_nf, oculta_nf),
                                    nn.Tanh())

        self.phi_h = nn.Sequential(nn.Linear(oculta_nf + 1, oculta_nf),
                                    nn.Tanh(),
                                    nn.Linear(oculta_nf, saida_nf))
        
        self.phi_v = nn.Sequential(nn.Linear(entrada_nf, oculta_nf),
                                    nn.Tanh(),
                                    nn.Linear(oculta_nf, 1))
        

    def distancia_radial(self, arestas, x):
        linha, col = arestas
        x_linha = x[linha]
        x_col = x[col]
        diferenca = x_linha - x_col
        dif_radial = t.sum((diferenca)**2, 1).unsqueeze(1)

        return dif_radial, diferenca
    
    def media_segmentada(self, arestas, x):
        max_index = max(arestas[0]) + 1
        resultado = [0] * max_index
        contagem = [0] * max_index
        for i, j in zip(arestas[0], arestas[1]):
            resultado[i] += t.sum(x[j]).item()  # Assumindo que x é um tensor e queremos o valor escalar
            contagem[i] += 1
        # Calculando a média, evitando divisão por zero
        media = [resultado[i] / contagem[i] if contagem[i] > 0 else 0 for i in range(max_index)]
        return t.tensor(media).unsqueeze(1)  # Convertendo de volta para tensor

    def soma_segmentada(self, arestas, x):
        max_index = max(arestas[0]) + 1
        resultado = [0] * max_index
        for i, j in zip(arestas[0], arestas[1]):
            resultado[i] += t.sum(x[j]).item()  # Assumindo que x é um tensor e queremos o valor escalar
        return t.tensor(resultado).unsqueeze(1)  # Convertendo de volta para tensor

    def forward(self, h, x, arestas, velocidade, atributos_arestas):
        linhas, cols = arestas
        h_linha = h[linhas]
        h_col = h[cols]
        dif_radial, diferenca = self.distancia_radial(arestas, x)

        # concatenação dos atributos
        atributos = t.cat((h_linha, h_col, dif_radial, atributos_arestas), 1)

        phi_v = self.phi_v(h)
        m_ij = self.phi_e(atributos)
        phi_x = self.phi_x(m_ij)
        velocidade = velocidade * phi_v + self.media_segmentada(arestas, diferenca * phi_x)
        x = x + velocidade
        m_i = self.soma_segmentada(arestas, m_ij)
        h = self.phi_h(t.cat((h, m_i), dim = 1))

        return h, x, velocidade

def get_edges(self, batch_size, n_nodes):
    edges = [t.LongTensor(self.edges[0]), t.LongTensor(self.edges[1])]
    if batch_size == 1:
        return edges
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [t.cat(rows), t.cat(cols)]
    return edges

class ModeloEquivariante(nn.Module):
    
    def __init__(self, entrada_nf: int, oculta_nf: int, saida_nf: int,
                    ij: int, n_camadas: int):
        """ Modelo de rede neural equivariante.

        :param entrada_nf: número de features de entrada de h
        :type entrada_nf: int
        :param oculta_nf: número de neurônios na camada oculta
        :type oculta_nf: int
        :param saida_nf: número de features de saída de x
        :type saida_nf: int
        :param ij: dimensão do vetor de atributos das arestas
        :type ij: int
        :param n_camadas: número de camadas da rede
        :type n_camadas: int
        """        
        super(ModeloEquivariante, self).__init__()
        self.entrada_nf = entrada_nf
        self.oculta_nf = oculta_nf
        self.saida_nf = saida_nf
        self.ij = ij
        self.n_camadas = n_camadas
        self.embedding_in = nn.Linear(entrada_nf, oculta_nf)
        for camada in range(n_camadas):
            self.add_module(f'camada_{camada}', CamadaEquivariante(oculta_nf, oculta_nf, oculta_nf, ij))
        self.to("cpu")
        self.embedding_out = nn.Linear(oculta_nf, saida_nf)

    def forward(self, h, x, arestas, velocidade, atributos_arestas):
        h = self.embedding_in(h)
        for camada in range(self.n_camadas):
            h, x, velocidade = self._modules[f'camada_{camada}'](h, x, arestas, velocidade, atributos_arestas)
        h = self.embedding_out(h)
        return h, x
    
device = "cpu"
loss_mse = nn.MSELoss()

def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 1, 'coord_reg': 1, 'counter': 1}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]
        loc, vel, edge_attr, _, loc_end = data

        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]

        optimizer.zero_grad()

        nodes = t.sqrt(t.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
        rows, cols = edges
        loc_dist = t.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
        edge_attr = t.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
        h, loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr)

        loss = loss_mse(loc_pred, loc_end)
        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size

    return res['loss'] / res['counter']


if __name__ == '__main__':
    # Número de Layers fixo pois foi descrito no artigo
    n_layers = 3
    lr_list = [1e-3, 1e-4, 1e-5]
    n_hidden_list = [64, 128]
    wd_list = [1e-8, 1e-10, 1e-12]
    best_val_loss = 0
    best_test_loss = 1e8
    best_epoch = 0
    n_epochs = 10
    best_lr = 0
    best_n_hidden = 0
    best_wd = 0
    for lr in lr_list:
        for n_hidden in n_hidden_list:
            for wd in wd_list:
                print(f"lr: {lr} \t n_hidden: {n_hidden} \t wd: {wd}")
                #loaders
                dataset_train = NBodyDataset(partition='train', dataset_name="nbody_small",
                                 max_samples=10000)
                loader_train = t.utils.data.DataLoader(dataset_train, batch_size=100, shuffle=True, drop_last=True)
        
                dataset_val = NBodyDataset(partition='val', dataset_name="nbody_small",
                                        max_samples=2000)
                loader_val = t.utils.data.DataLoader(dataset_val, batch_size=100, shuffle=True, drop_last=True)

                dataset_test = NBodyDataset(partition='test', dataset_name="nbody_small",
                                            max_samples= 2000)
                loader_test = t.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=True, drop_last=True)

                model = ModeloEquivariante(1, n_hidden, 3, 2, n_layers)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                results = { 'lr': lr,
                            'n_hidden': n_hidden,
                            'wd': wd,
                            'epochs': [],
                            'losess_train': [],
                            'losess_val': []}
                for epoch in range(0, n_epochs):
                    train_loss = train(model, optimizer, epoch, loader_train)
                    print("Epoch %d \t Train Loss: %.5f" % (epoch, train_loss))
                    results['epochs'].append(epoch)
                    results['losess_train'].append(train_loss)
                    val_loss = train(model, optimizer, epoch, loader_val, backprop=False)
                    print("Epoch %d \t Val Loss: %.5f" % (epoch, val_loss))
                    results['losess_val'].append(val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        best_lr = lr
                        best_n_hidden = n_hidden
                        best_wd = wd
                json_object = json.dumps(results, indent=4)
                with open(F"data/losess_lr_{lr}_n_hidden_{n_hidden}_wd_{wd}.json", "w") as outfile:
                    outfile.write(json_object)

    dataset_train = NBodyDataset(partition='train', dataset_name="nbody_small",
                                 max_samples=10000)
    loader_train = t.utils.data.DataLoader(dataset_train, batch_size=100, shuffle=True, drop_last=True)

    dataset_val = NBodyDataset(partition='val', dataset_name="nbody_small",
                            max_samples=2000)
    loader_val = t.utils.data.DataLoader(dataset_val, batch_size=100, shuffle=True, drop_last=True)

    dataset_test = NBodyDataset(partition='test', dataset_name="nbody_small",
                                max_samples= 2000)
    loader_test = t.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=True, drop_last=True)
    print(f"Melhor loss: {best_val_loss}",
            f"Melhor época: {best_epoch}",
            f"Melhor lr: {best_lr}",
            f"Melhor n_hidden: {best_n_hidden}",
            f"Melhor wd: {best_wd}")
    best_model = ModeloEquivariante(1, best_n_hidden, 3, 2, n_layers)
    optimizer = optim.Adam(best_model.parameters(), lr=best_lr, weight_decay=best_wd)
    results = { 'lr': best_lr,
                'n_hidden': best_n_hidden,
                'wd': best_wd,
                'epochs': [],
                'losess_train': [],
                'losess_val': [],
                'losess_test': []}
    for epoch in range(0, best_epoch):
        results['epochs'].append(epoch)
        train_loss = train(best_model, optimizer, epoch, loader_train)
        results['losess_train'].append(train_loss)
        print("Epoch %d \t Train Loss: %.5f" % (epoch, train_loss))
        val_loss = train(best_model, optimizer, epoch, loader_val, backprop=False)
        results['losess_val'].append(val_loss)
        print("Epoch %d \t Val Loss: %.5f" % (epoch, val_loss))
        test_loss = train(best_model, optimizer, epoch, loader_test, backprop=False)
        results['losess_test'].append(test_loss)
        print("Epoch %d \t Test Loss: %.5f" % (epoch, test_loss))

    json_object = json.dumps(results, indent=4)
    with open(F"data/losess_best_model.json", "w") as outfile:
        outfile.write(json_object)

    

