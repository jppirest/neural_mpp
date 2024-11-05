import torch.nn as nn
import torch.nn.functional as F
import torch
from models import NormalizingFlow

class GrangerMPP(nn.Module):

    def __init__(self, Process,
    num_features:int = 1, 
    hidden_dim : int = 32, 
    num_layers : int = 4
    ):

        super().__init__()

        self.Process = Process
        self.n_processes = self.Process.n_processes
        self.memory_dim = self.Process.memory_dim
        self.GrangerMatrix = nn.Parameter((torch.empty(self.n_processes, self.n_processes)))
        nn.init.normal_(self.GrangerMatrix, mean=0.5, std=0.1) 

        self.models = nn.ModuleList([NormalizingFlow(num_features = num_features, memory_size = self.memory_dim, hidden_dim = hidden_dim, num_layers = num_layers) for i in range(self.n_processes)])
        self.optimizers = [torch.optim.Adam(list(self.models[i].parameters()), lr=1e-4, weight_decay = 1e-5) for i in range(self.n_processes)]
        self.g_optimizer = torch.optim.Adam([self.GrangerMatrix], lr = 1e-3, weight_decay=1e-5)
        self.log_GrangerMatrix = []
        self.sweep_dict = self.Process.make_dict()


    def em_step(self, n_steps, tau0 : float = 2.0, tau1 : float = 1.0):

        assert (tau0 > tau1, 'Initial Gumbel-Softmax Temperature should be higher than final temperature.')
        
        dic = {}
        self.causes = [[] for _ in range(self.n_processes)]
        
        for i in range(self.n_processes):
            dic[i] = []


        taus = torch.linspace(tau0, tau1, steps = n_steps)
        
        for self.step in range(n_steps):
          for i_proc in range(self.n_processes):
              self.causes[i_proc] = []
              curr = self.Process.processes[i_proc]
              len_curr = len(curr)
              idx_start = 0
              
              # Iterating through the process using a while loop:
              
              while idx_start < len_curr:
                self.num_events = 5
                events = self.Process.get_events(self.sweep_dict, i_proc, idx_start, self.num_events) #DataLoader type
                self.events = events
                
                # Acessing the unique element of the dataloader.
                
                for _, (X, plausible_cause, time_idx) in enumerate(events):
                    
                    possible_times =  time_idx.unique()
                    app = []
                    
                    for _, val in enumerate(possible_times):
                        where_ = torch.where(time_idx == val)[0]
                        app.append(plausible_cause[where_])
                    

                    gumbel = self.sample(self.num_events, i_proc, tau = taus[self.step])
                    
                    cause_rank_list = []
                    
                    # Renormalizing the probability for a given time instant.
                    
                    for pos, values in enumerate(app):
                        curr_probabilities = gumbel[pos]
                        num_of_plausible_causes = len(values)
                        if num_of_plausible_causes != 1:
                            renormalized_ = curr_probabilities[values]
                            renormalized_ = renormalized_/renormalized_.sum()
                        else:
                            renormalized_ = torch.tensor(1)
                        cause_rank_list.append(renormalized_)
                    
                    
                    cause_rank_tensor = torch.cat([t.unsqueeze(0) if t.dim() == 0 else t for t in cause_rank_list])
                    
                    self.cause_rank_tensor = cause_rank_tensor
                    self.causes[i_proc].append((cause_rank_tensor, plausible_cause))
                    
                    # Normalizing by the len of the current process and its target.

                    normalizing_len = 1/len_curr * torch.tensor([len(self.Process.processes[i]) for i in plausible_cause])
                    
                    
                    X = X.unsqueeze(-1)
                    loss = self.m_step(i_proc, X, self.cause_rank_tensor, normalizing_len)
                    dic[i_proc].append(loss)

                    idx_start += self.num_events

              if (self.step + 1) % 5 == 0 or self.step == 0:
                  print(f'Step: {self.step + 1}, Model: {i_proc}, Loss: {loss}')


        return dic

    def m_step(self, i_proc, X, cause_rank_tensor, normalizing_len):
        
        model = self.models[i_proc]
        self.optimizers[i_proc].zero_grad()
        self.g_optimizer.zero_grad()
        batch_size, memory, features = X.size()
        z, logp = model.log_prob(X)
        loss = -1*logp

        loss_rnn = (normalizing_len * loss * cause_rank_tensor).sum()/batch_size  + -1*(normalizing_len*torch.log(cause_rank_tensor + 1e-7)).sum()/batch_size + 0.001*self.GrangerMatrix[i_proc].norm(p=1)


        if not (torch.isnan(loss_rnn) | torch.isinf(loss_rnn)):

            self.log_GrangerMatrix.append(self.GrangerMatrix.clone())
            
            loss_rnn.backward(retain_graph = True)

            self.optimizers[i_proc].step()
            self.g_optimizer.step()
            self.log_GrangerMatrix.append(self.GrangerMatrix.clone().detach())


        else:
            print(f'NaN found in epoch: {self.step}')

        return loss_rnn.item()

    def sample(self, num_events, i_proc, tau):

      in_ = self.GrangerMatrix[i_proc]
      rv = []
      for i in range(num_events):
        cause = F.gumbel_softmax(
            in_,
            tau = tau,
            hard = False
        )
        rv.append(cause)

    #   self.causes[i_proc].append(rv)

      return rv


