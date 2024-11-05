import torch
from torch.utils.data import DataLoader


class Memory:

  def __init__(self, size):
    self.size = size
    self.stack = []

  def push(self, item):
    if len(self.stack) >= self.size:
        self.pop()
    self.stack.append(item)

  def pop(self):
    if self.stack:
      return self.stack.pop(0)
    else:
      return None

  def __repr__(self):
    return repr(self.stack)
  
  def copy(self):
    return self.stack.copy()
  
  def __iter__(self):
    return iter(self.stack[::-1])
  
  def full(self):
    return len(self.stack) == self.size

class Sweep():
  def __init__(self, processes, memory_dim):
    self.processes = processes
    self.n_processes = len(processes)
    self.memory_dim = memory_dim

  def make_dict(self):
    pass

  def get_events(self, sweep_dict, base_process, idx_start, num_events):
    curr_process = sweep_dict[base_process]
    
    if idx_start + num_events > len(self.processes[base_process]):
        num_events = len(self.processes[base_process]) - idx_start
    
    app = [] 
    for idx in range(idx_start, idx_start + num_events):
        for cause in range(self.n_processes):
            event = curr_process[cause][idx]
            if -1 not in event:
                app.append((event, cause))

    return DataLoader(app, shuffle = False, batch_size = len(app))


class HawkesSweep(Sweep):

  def make_dict(self):
    dic = {}
    for i in range(self.n_processes):
        target = self.processes[i]
        dic[i] = {}
        for j in range(self.n_processes):
            cause = self.processes[j]
            dic[i][j] = self.sweep(target, cause)

    return dic

  def sweep(self, pa, pc):
    events = []
    pa_indices = []
    for i, ia in enumerate(pa):
        events.append((ia, 'a'))
        pa_indices.append(i)

    for ic in pc:
        events.append((ic, 'c'))

    lim = self.memory_dim
    events.sort()
    mem = Memory(lim)
    ret = []
    for t, e in events:
        if e == 'c':
            mem.push(t)

        if e == 'a':
            if not mem.full():
                pp = [-1] * lim
            else:
                pp = [t - tc for tc in mem]
                ret.append(pp)
    
    return torch.tensor(ret, dtype=torch.float)


class WoldSweep(Sweep):
    def construct_wold_dict(self):
        dict = {}

        events = []
        for id, process in enumerate(self.processes):
            for t in process:
                events.append((t.item(), id))

        events.sort()

        deltas = {}
        last = {}
        cur = -1

        for t, id in events:
            dict[t] = {}
            deltas[id] = Memory(self.memory_dim)
            last[id] = [0, 0]

        for t, id in events:
            if t != cur:
                
                # updating
                cur = t
                for _id, _delta in deltas.items():
                    dict[cur][_id] = _delta.copy()
            
            last[id][1] = last[id][0]
            last[id][0] = t
            if last[id][1] != 0:
                deltas[id].push(last[id][0] - last[id][1])
            
        return dict
        
    # TODO: check idx_start semantics
    def make_dict(self):
        wold = self.construct_wold_dict()
        dic = {}
        for i in range(self.n_processes):
            target = self.processes[i]
            dic[i] = {}
            #for j in range(self.n_processes):
            #    cause = self.processes[j]
            #    dic[i][j] = self.sweep(target, cause)
            ret = {}
            for _t in target:
                t = _t.item()
                for j in range(self.n_processes):
                    if j not in ret:
                        ret[j] = []
                    #print(t, j, wold[t][j])
                    #return None
                    if len(wold[t][j]) < self.memory_dim:
                        ret[j].append([-1] * self.memory_dim)
                    else:
                        ret[j].append(wold[t][j])

            #return None
            for j in range(self.n_processes):
                dic[i][j] = torch.tensor(ret[j], dtype=torch.float)

        return dic
