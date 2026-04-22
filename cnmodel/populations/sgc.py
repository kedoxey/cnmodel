import logging
import random
import numpy as np
# import pyqtgraph.multiprocess as mp
import multiprocessing as mp

from .population import Population
from .. import cells
from ..util import sound

class SGC(Population):
    """A population of spiral ganglion cells.
    
    The cell distribution is uniform from 2kHz to 64kHz, evenly divided between
    spontaneous rate groups.
    """
    type = 'sgc'
    
    def __init__(self, species='mouse', model='dummy', **kwds):
        # Completely fabricated cell distribution: uniform from 2kHz to 40kHz,
        # evenly divided between SR groups. We only go up to 40kHz because the
        # auditory periphery model does not support >40kHz.
        freqs = self._get_cf_array(species)
        fields = [
            ('cf', float),
            ('sr', int),  # 0=low sr, 1=mid sr, 2=high sr
        ]
        super(SGC, self).__init__(species, len(freqs), fields=fields, model=model, **kwds)
        self._cells['cf'] = freqs
        # old version:
            # evenly distribute SR groups
            #self._cells['sr'] = np.arange(len(freqs)) % 3
        # new version:
        # draw from distributions matching approximate SR distribution 
        sr_vals = self.get_sgcsr_array(freqs, species='mouse')
        self._cells['sr'] = sr_vals
        
    def set_seed(self, seed):
        self.next_seed = seed
    
    def create_cell(self, cell_rec):
        """ Return a single new cell to be used in this population. The 
        *cell_rec* argument is the row from self.cells that describes the cell 
        to be created.
        """
        return cells.SGC.create(species=self.species, cf=cell_rec['cf'],
                                sr=cell_rec['sr'], **self._cell_args)
        
    def connect_pop_to_cell(self, pop, index):
        # SGC does not support any inputs
        assert len(self.connections) == 0

    def parallel_spiketrains(self, stim, seed, ind, train):
        cell = self.get_cell(ind)
        spiketrain = cell.generate_spiketrain(stim, seed)
        train.extend([spkt for spkt in spiketrain])
        # return spiketrain

    def get_sgc_lost_array(self, cell_ids, loss_frac):
        """ Return list of SGC cell ids that are to be "removed" due to high-
        frequency hearing loss.
        """

        lost_cells = []
        for cell_id in cell_ids:
            cell = self.get_cell(cell_id)
            if cell._cf > self._loss_limit:
                lost_cells.append(cell_id)

        loss_frac = self._loss_frac / 100
        ind_remove = set(random.sample(list(range(len(lost_cells))), int(loss_frac*len(lost_cells))))
        lost_cells = [n for i, n in enumerate(lost_cells) if i in ind_remove]
        
        return lost_cells

    def set_sound_stim(self, stim, parallel=False):
        """Set a sound stimulus to generate spike trains for all (real) cells
        in this population.
        """
        real = self.real_cells()
        lost_reals = self.get_sgc_lost_array(real, self._loss_frac)
        logging.info("Assigning spike trains to %d SGC cells..", len(real))
        if not parallel:
            for i, ind in enumerate(real):
                #logging.info("Assigning spike train to SGC %d (%d/%d)", ind, i, len(real))
                cell = self.get_cell(ind)
                cell_hearing = 'normal'
                cell_lost = False
                if (cell.cf > self._loss_limit) and ('loss' in self._hearing):  # Method 1
                    cell_hearing = 'loss'
                if (ind in lost_reals) and ('loss' in self._hearing):  # Method 3
                    cell_lost = True
                cell.set_sound_stim(stim, self.next_seed, hearing=cell_hearing, cell_lost=cell_lost)
                self.next_seed += 1

        else:
            seeds = range(self.next_seed, self.next_seed + len(real))
            self.next_seed = seeds[-1] + 1
            tasks = [(s,r) for s,r in zip(seeds, real)]
            # trains = [None] * len(tasks)
            # generate spike trains in parallel
            # with mp.Parallelize(enumerate(tasks), trains=trains, progressDialog='Generating SGC spike trains..') as tasker:
            #     for i, x in tasker:
            #         seed, ind = x
            #         cell = self.get_cell(ind)
            #         train = cell.generate_spiketrain(stim, seed)
            #         tasker.trains[i] = train

            trains = []
            for seed, ind in tasks:
                train = mp.Manager().list()
                p1 = mp.Process(target=self.parallel_spiketrains, args=(stim, seed, ind, train))
                p1.start()
                p1.join()
                trains.append(list(train))
            # collected all trains; now assign to cells
            for i,ind in enumerate(real):
                cell = self.get_cell(ind)
                cell.set_spiketrain(trains[i])
    
    


