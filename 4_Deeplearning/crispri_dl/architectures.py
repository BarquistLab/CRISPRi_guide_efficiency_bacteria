############################################
# imports
############################################

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torchmetrics.functional import mean_squared_error

############################################
# 1D CNN class
############################################

class Crispr1DCNN(pl.LightningModule):

    def __init__(self, num_features, which_loss = "mse"):
        super().__init__()
     
        self.margin = 0.1
        self.alpha = 100
        self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.which_loss = which_loss

        n = 32
        latent = 32
        self.cnn1 = nn.Sequential(nn.Conv1d(4, 2 * n, kernel_size=5, stride=2), nn.ReLU() , nn.Conv1d(2 * n, 2 * n, kernel_size=3, stride=2), nn.ReLU(), nn.Conv1d(2 * n, latent, kernel_size=1), nn.ReLU(), nn.Flatten())
        self.cnn2 = nn.Sequential(nn.Conv1d(4, 2 * n, kernel_size=5, stride=2), nn.ReLU(),  nn.Conv1d(2 * n, 2 * n, kernel_size=3, stride=2), nn.ReLU(), nn.Conv1d(2 * n, latent, kernel_size=1), nn.ReLU(), nn.Flatten())

        self.ln1 = nn.Linear(num_features+latent*6, 128)
        self.ln2 = nn.Linear(128, 64)
        self.ln3 = nn.Linear(64, 32)
        

        if which_loss == "mse":
            self.ln4 = nn.Linear(32, 1)

        elif which_loss == "pinball":
            self.ln4 = nn.Linear(32, len(self.quantiles))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)            
        self.bn1=nn.BatchNorm1d(128)
        self.bn2=nn.BatchNorm1d(64)
        self.bn3=nn.BatchNorm1d(32)

        
    def forward(self,  x_sequence_30nt, x_features):

        # reorder dimensions to fit required input
        #x_sequence_20nt = x_sequence_20nt.permute(0, 2, 1)
        x_sequence_30nt = x_sequence_30nt.permute(0, 2, 1)
        #print("-----")
        #print(x_sequence_40nt.size())
        #print(x_features.size())
        # 1D CNN layer
        #cnn_sequence_20nt = self.cnn1(x_sequence_20nt)
        cnn_sequence_40nt = self.cnn2(x_sequence_30nt)

        #print(cnn_sequence_20nt.size())
        #print("*****")
        # linear layers
        #lin_sequence_20nt = self.relu(self.ln1(cnn_sequence_20nt))
        #lin_sequence_30nt = self.relu(self.ln2(cnn_sequence_30nt))
        #lin_pam = self.relu(self.ln3(torch.flatten(x_pam, start_dim=1)))
        #lin_features = self.relu(self.ln4(torch.flatten(x_features, start_dim=1)))
        #print(x_features.size())
        # concat all layers
        #concat_layer = torch.cat((lin_features, lin_pam, lin_sequence_20nt, lin_sequence_30nt), 1)
        concat_layer = torch.cat((x_features, cnn_sequence_40nt), 1)
        
        out = self.dropout(self.relu(self.bn1(self.ln1(concat_layer))))
        out = self.dropout(self.relu(self.bn2(self.ln2(out))))
        out = self.dropout(self.relu(self.bn3(self.ln3(out))))
        out = self.ln4(out)
                                                                                          
        return out


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


    def quantile_loss(self, predictions, target):
        assert not target.requires_grad
        assert predictions.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - predictions[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        quantile_loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
          
        diff = predictions[:, 1:] - predictions[:, :-1]
        penalty = torch.mean(self.relu(self.margin - diff)) * self.alpha
        return quantile_loss + penalty


    def training_step(self, batch, batch_idx):
        x_sequence_30nt, x_features, y = batch
        y = torch.unsqueeze(y, 1)
        y_pred = self.forward( x_sequence_30nt, x_features)
        if self.which_loss == "mse":
          loss = F.mse_loss(y_pred, y)
        elif self.which_loss == "pinball":
          loss = self.quantile_loss(y_pred, y)
        return {'loss': loss}


    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=True)


    # def validation_step(self, batch, batch_idx):
    #     x_sequence_30nt, x_features, y = batch
    #     y = torch.unsqueeze(y, 1)
    #     y_pred = self.forward(x_sequence_30nt, x_features)
    #     if self.which_loss == "mse":
    #       loss = F.mse_loss(y_pred, y)
    #     elif self.which_loss == "pinball":
    #       loss = self.quantile_loss(y_pred, y)
    #     return {'val_loss': loss}
    def validation_step(self, batch, batch_idx):
        x_sequence_30nt, x_features, y = batch
        y = torch.unsqueeze(y, 1)
        y_pred = self(x_sequence_30nt, x_features)
      
        loss = mean_squared_error(y_pred, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)
        

    # def test_step(self, batch, batch_idx):
    #     x_sequence_40nt, x_features, y = batch
    #     y = torch.unsqueeze(y, 1)
    #     y_pred = self.forward(x_sequence_40nt, x_features)
    #     if self.which_loss == "mse":
    #       loss = F.mse_loss(y_pred, y)
    #     elif self.which_loss == "pinball":
    #       loss = self.quantile_loss(y_pred, y)
    #     return {'test_loss': loss}
    def test_step(self, batch, batch_idx):
        x_sequence_30nt, x_features, y = batch
        y = torch.unsqueeze(y, 1)
        y_pred = self(x_sequence_30nt, x_features)
        loss = mean_squared_error(y_pred, y)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss, prog_bar=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_sequence_30nt, x_features, y = batch
        y = torch.unsqueeze(y, 1)
        y_pred = self(x_sequence_30nt, x_features)
        return y_pred

############################################
# GRU class
############################################
        
        
class CrisprGRU(pl.LightningModule):

    def __init__(self, num_features, which_loss = "mse"):
        super().__init__()
     
        self.margin = 0.1
        self.alpha = 100
        self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.which_loss = which_loss

        #self.gru1 = nn.GRU(4, 100,1, bidirectional=True)
        self.gru2 = nn.GRU(4, 50,2, bidirectional=True,dropout=0.5)

        self.ln1 = nn.Linear(100+num_features, 128)
        self.ln2 = nn.Linear(128, 64)
        self.ln3 = nn.Linear(64, 32)
        

        if which_loss == "mse":
          self.ln4 = nn.Linear(32, 1)

        elif which_loss == "pinball":
          self.ln4 = nn.Linear(32, len(self.quantiles))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
            
        self.bn1=nn.BatchNorm1d(128)
        self.bn2=nn.BatchNorm1d(64)
        self.bn3=nn.BatchNorm1d(32)

        
    def forward(self, x_sequence_30nt, x_features):

        # reorder dimensions to fit required input
        #x_sequence_20nt = x_sequence_20nt.permute(1, 0, 2)
        x_sequence_30nt = x_sequence_30nt.permute(1, 0, 2)
        
        # GRU layer
        #_, gru_sequence_20nt = self.gru1(x_sequence_20nt)
        _, gru_sequence_40nt = self.gru2(x_sequence_30nt)

        #gru_sequence_20nt = torch.cat([gru_sequence_20nt[0,:, :], gru_sequence_20nt[1,:,:]], dim=1)
        gru_sequence_40nt = torch.cat([gru_sequence_40nt[0,:, :], gru_sequence_40nt[1,:,:]], dim=1)

        # linear layers
        #lin_sequence_20nt = self.relu(self.ln1(gru_sequence_20nt.squeeze(0)))
        #lin_sequence_30nt = self.relu(self.ln2(gru_sequence_30nt.squeeze(0)))
        #lin_pam = self.relu(self.ln3(torch.flatten(x_pam, start_dim=1)))
        #lin_features = self.relu(self.ln4(torch.flatten(x_features, start_dim=1)))

        #print("lin_sequence_20nt" + str(lin_sequence_20nt.size()))
        #print("lin_sequence_30nt" + str(lin_sequence_30nt.size()))
        #print("lin_pam" + str(lin_pam.size()))
        #print("lin_features" + str(lin_features.size()))

        # concat all layers
        #concat_layer = torch.cat((lin_features, lin_pam, lin_sequence_20nt, lin_sequence_30nt), 1)
            
        out = torch.cat((x_features, gru_sequence_40nt), 1)
        out = self.dropout(self.relu(self.bn1(self.ln1(out))))
        out = self.dropout(self.relu(self.bn2(self.ln2(out))))
        out = self.dropout(self.relu(self.bn3(self.ln3(out))))
        out = self.ln4(out)
        
        return out
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


    def quantile_loss(self, predictions, target):
        assert not target.requires_grad
        assert predictions.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - predictions[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        quantile_loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
          
        diff = predictions[:, 1:] - predictions[:, :-1]
        penalty = torch.mean(self.relu(self.margin - diff)) * self.alpha
        return quantile_loss + penalty


    def training_step(self, batch, batch_idx):
        x_sequence_30nt, x_features, y = batch
        y = torch.unsqueeze(y, 1)
        y_pred = self.forward(x_sequence_30nt, x_features)
        if self.which_loss == "mse":
          loss = F.mse_loss(y_pred, y)
        elif self.which_loss == "pinball":
          loss = self.quantile_loss(y_pred, y)
        return {'loss': loss}


    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=True)


    # def validation_step(self, batch, batch_idx):
    #     x_sequence_30nt, x_features, y = batch
    #     y = torch.unsqueeze(y, 1)
    #     y_pred = self.forward(x_sequence_30nt, x_features)
    #     if self.which_loss == "mse":
    #       loss = F.mse_loss(y_pred, y)
    #     elif self.which_loss == "pinball":
    #       loss = self.quantile_loss(y_pred, y)
    #     return {'val_loss': loss}
    def validation_step(self, batch, batch_idx):
        x_sequence_30nt, x_features, y = batch
        y = torch.unsqueeze(y, 1)
        y_pred = self(x_sequence_30nt, x_features)
      
        loss = mean_squared_error(y_pred, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)
        

    # def test_step(self, batch, batch_idx):
    #     x_sequence_40nt, x_features, y = batch
    #     y = torch.unsqueeze(y, 1)
    #     y_pred = self.forward(x_sequence_40nt, x_features)
    #     if self.which_loss == "mse":
    #       loss = F.mse_loss(y_pred, y)
    #     elif self.which_loss == "pinball":
    #       loss = self.quantile_loss(y_pred, y)
    #     return {'test_loss': loss}
    def test_step(self, batch, batch_idx):
        x_sequence_30nt, x_features, y = batch
        y = torch.unsqueeze(y, 1)
        y_pred = self(x_sequence_30nt, x_features)
        loss = mean_squared_error(y_pred, y)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss, prog_bar=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_sequence_30nt, x_features, y = batch
        y = torch.unsqueeze(y, 1)
        y_pred = self(x_sequence_30nt, x_features)
        return y_pred
        
        
############################################
# CRISPRON
############################################

class CrisprOn1DCNN(pl.LightningModule):

    def __init__(self, num_features = 2790):
        super().__init__()
     
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 100, 3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AvgPool1d(2),
            nn.Flatten(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 70, 5),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AvgPool1d(2),
            nn.Flatten(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(4, 40, 7),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AvgPool1d(2),
            nn.Flatten(),
        )

        self.dense_layers1 = nn.Sequential(
            nn.Linear(2790, 80),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.dense_layers2 = nn.Sequential(
            nn.Linear(88, 80),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(80, 60),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(60, 1),
        )        
        
    def forward(self,  x_sequence_30nt, x_features):

        # reorder dimensions to fit required input
        #x_sequence_20nt = x_sequence_20nt.permute(0, 2, 1)
        x_sequence_30nt = x_sequence_30nt.permute(0, 2, 1)
        
        dense_inputs = []
        dense_inputs.append(self.conv1(x_sequence_30nt))
        dense_inputs.append(self.conv2(x_sequence_30nt))
        dense_inputs.append(self.conv3(x_sequence_30nt))
        concat_tensors = torch.cat(dense_inputs, 1)
       
       
       # feed through dense layers
        dense_output1 = self.dense_layers1(concat_tensors)

        # concatenate biofeature
        concat_tensors_with_biofeature = torch.cat((dense_output1, x_features), 1)
        out = self.dense_layers2(concat_tensors_with_biofeature)
                                                                                          
        return out


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)        
        return optimizer


    def training_step(self, batch, batch_idx):
        x_sequence_30nt, x_features, y = batch
        y = torch.unsqueeze(y, 1)
        y_pred = self.forward( x_sequence_30nt, x_features)
       
        loss = mean_squared_error(y_pred, y)

        return loss


    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        x_sequence_30nt, x_features, y = batch
        y = torch.unsqueeze(y, 1)
        y_pred = self(x_sequence_30nt, x_features)
      
        loss = mean_squared_error(y_pred, y)
        return {'val_loss': loss}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)
        

    def test_step(self, batch, batch_idx):
        x_sequence_30nt, x_features, y = batch
        y = torch.unsqueeze(y, 1)
        y_pred = self(x_sequence_30nt, x_features)
        loss = mean_squared_error(y_pred, y)
        return {'test_loss': loss}


    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss, prog_bar=True)



    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_sequence_30nt, x_features, y = batch
        y = torch.unsqueeze(y, 1)
        y_pred = self(x_sequence_30nt, x_features)
        return y_pred
    