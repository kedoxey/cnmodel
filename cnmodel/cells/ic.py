from __future__ import print_function
from neuron import h
from collections import OrderedDict
from .cell import Cell
from .. import synapses
from ..util import nstomho
from ..util import Params
import numpy as np
from .. import data
import pprint
pp = pprint.PrettyPrinter(indent=4, width=60)
    
__all__ = ['IC', 'ICDefault']


class IC(Cell):
    
    celltype = 'ic'
    scaled = False  # only allow scaling once
    
    @classmethod
    def create(cls, model='RM03', **kwds):
        if model == 'RM03':
            return ICDefault(**kwds)
        else:
            raise ValueError ('IC model %s is unknown', model)
        
    def __init__(self):
        Cell.__init__(self)
        self.spike_source = None

    def make_psd(self, terminal, psd_type, **kwds):
        """
        Connect a presynaptic terminal to one post section at the specified location, with the fraction
        of the "standard" conductance determined by gbar.
        The default condition is designed to pass the unit test (loc=0.5)
        
        Parameters
        ----------
        terminal : Presynaptic terminal (NEURON object)
        
        psd_type : only simple for ic cell
        
        kwds: dictionary of options. 
            Two are currently handled:
            postsite : expect a list consisting of [sectionno, location (float)]
            AMPAScale : float to scale the ampa currents
        
        """
        if 'postsite' in kwds:  # use a defined location instead of the default (soma(0.5)
            postsite = kwds['postsite']
            loc = postsite[1]  # where on the section?
            uname = 'sections[%d]' % postsite[0]  # make a name to look up the neuron section object
            post_sec = self.hr.get_section(uname)  # Tell us where to put the synapse.
        else:
            loc = 0.5
            post_sec = self.soma
        
        if psd_type == 'simple':
            # presynaptic terminal
            if terminal.cell.celltype in ['pyramidal']:
                weight = data.get('%s_synapse' % terminal.cell.celltype, species=self.species,
                        post_type=self.celltype, field='weight')
                tau1 = data.get('%s_synapse' % terminal.cell.celltype, species=self.species,
                        post_type=self.celltype, field='tau1')
                tau2 = data.get('%s_synapse' % terminal.cell.celltype, species=self.species,
                        post_type=self.celltype, field='tau2')
                erev = data.get('%s_synapse' % terminal.cell.celltype, species=self.species,
                        post_type=self.celltype, field='erev')
                return self.make_exp2_psd(post_sec, terminal, weight=weight, loc=loc,
                        tau1=tau1, tau2=tau2, erev=erev)
            else:
                raise TypeError("Cannot make simple PSD for %s => %s" % 
                            (terminal.cell.celltype, self.celltype))
            
        elif psd_type == 'multisite':
            raise ValueError('Multisite synapse not supported')
        else:
            raise ValueError("Unsupported psd type %s" % psd_type)

    def make_terminal(self, post_cell, term_type, **kwds):
        if term_type == 'simple':
            return synapses.SimpleTerminal(self.soma, post_cell, **kwds)

        elif term_type == 'multisite':
            raise ValueError('Mulitsite terminal not supported')
        else:
            raise ValueError("Unsupported terminal type %s" % term_type)


class ICDefault(IC):
    """
    Inferior Colliculus base model.
    Rothman and Manis, 2003abc (Type I-c), for guinea pig (use mouse to match population parameters)
    """

    def __init__(self, morphology=None, decorator=None, nach=None,
                 ttx=False, species='mouse', modelType=None, modelName=None,
                 debug=False, temperature=None):
        """
        Create a bushy cell, using the default parameters for guinea pig from
        R&M2003, as a type II cell.
        Additional modifications to the cell can be made by calling methods below.
        
        Parameters
        ----------
        morphology : string (default: None)
            Name of a .hoc file representing the morphology. This file is used to constructe
            an electrotonic (cable) model. 
            If None (default), then a "point" (really, single cylinder) model is made, exactly according to RM03.
            
        decorator : Python function (default: None)
            decorator is a function that "decorates" the morphology with ion channels according
            to a set of rules.
            If None, a default set of channels is inserted into the first soma section, and the
            rest of the structure is "bare".
        
        nach : string (default: None)
            nach selects the type of sodium channel that will be used in the model. A channel mechanism
            by that name must exist. The default channel is set to 'nacn' (R&M03)
        
        temperature : float (default: 22)
            temperature to run the cell at. 
                 
        ttx : Boolean (default: False)
            If ttx is True, then the sodium channel conductance is set to 0 everywhere in the cell.
            This flag duplicates the effects of tetrodotoxin in the model. Currently, the flag is not implemented.
        
        species: string (default 'mouse')
            species defines the pattern of ion channel densities that will be inserted, according to 
            prior measurements in various species. Note that
            if a decorator function is specified, this argument is ignored as the decorator will
            specify the channel density.

        modelName: string (default: None)
            modelName specifies the source conductance pattern (RM03, XM13, etc).
            modelName is passed to the decorator, or to species_scaling to adjust point (single cylinder) models.
                             
        modelType: string (default: None)
            modelType specifies the subtype of the cell model that will be used (e.g., "II", "II-I", etc).
            modelType is passed to the decorator, or to species_scaling to adjust point (single cylinder) models.
            
        debug: boolean (default: False)
            When True, there will be multiple printouts of progress and parameters.
            
        Returns
        -------
            Nothing
                 
        """
        super(ICDefault, self).__init__()
        self.i_test_range={'pulse': (-0.15, 0.15, 0.01)}  # note that this might get reset with decorator according to channels
                                                    # The default values are set in the species_scaling routine
        if modelType == None:
            modelType = 'I-c'

        if species == 'mouse':
            modelName = 'RM03'
            temp = 22.
            dataset = 'RM03_channels'
        else:
            raise ValueError(f"Species {species:s} not recognized for {self.celltype:s} cells")

        self.debug = debug
        self.status = {'species': species, 'cellClass': self.celltype, 'modelType': modelType, 'modelName': modelName,
                       self.somaname: True, 'axon': False, 'dendrites': False, 'pumps': False, 'hillock': False, 
                       'initialsegment': False, 'myelinatedaxon': False, 'unmyelinatedaxon': False,
                       'na': nach, 'ttx': ttx, 'name': self.celltype,
                       'morphology': morphology, 'decorator': decorator, 'temperature': temperature}
        self.vrange = [-70., -55.]  # set a default vrange for searching for rmp

        self.c_m = 0.9  # default in units of uF/cm^2
        self.spike_threshold = -40
        if self.debug:
            print( 'model type, model name, species, somaname: ', modelType, modelName, species, nach, self.somaname)


        self._valid_temperatures = (temp, )
        if self.status['temperature'] == None:
            self.status['temperature'] = temp

        self.do_morphology(morphology)

        self.pars = self.get_cellpars(dataset, species=species, modelType=modelType)
        self.status['na'] = self.pars.natype

        # decorate the morphology with ion channels
        if decorator is None:   # basic "point" model, only on the soma, uses table data for soma.
            self.mechanisms = ['kht', 'ka', 'ihvcn', 'leak', self.pars.natype]
            for mech in self.mechanisms:
                self.soma.insert(mech)
            self.soma.ena = self.e_na
            self.soma.ek = self.e_k
            self.soma().ihvcn.eh = self.e_h
            self.soma().leak.erev = self.e_leak
            self.species_scaling(silent=True)  # set the default type II cell parameters
        else:  # decorate according to a defined set of rules on all cell compartments, with tables.
            self.decorate()

        self.save_all_mechs()  # save all mechanisms inserted, location and gbar values...
        self.get_mechs(self.soma)
        if debug:
            print (f"   << Created {self.celltype.title():s} cell >>")

    def get_cellpars(self, dataset, species='mouse', modelType='II'):
        """
        Retrieve parameters for the specifed model type and species from the data tables
        
        dataset : str (no default)
            name of the data table to use
        
        species : str (default: 'mouse')
            Species table to use
        
        modelType : str (default: 'II')
            Model type to get parameters from the table.
        
        """
        cellcap = data.get(dataset, species=species, model_type=modelType,
            field='soma_Cap')  # total somatic capacitance (point cells)
        chtype = data.get(dataset, species=species, model_type=modelType,
            field='na_type')
        pars = Params(cap=cellcap, natype=chtype)
        if self.status['modelName'] == 'RM03':
            for g in ['%s_gbar' % pars.natype, 'kht_gbar', 'klt_gbar', 'ih_gbar', 'leak_gbar']:
                pars.additem(g,  data.get(dataset, species=species, model_type=modelType,
                    field=g))
        else:
            raise ValueError(f"get_cellpars: Model name {self.status['modelName']} is not yet implemented for cell type {self.celltype.title():s}")

        if self.debug:
            pars.show()
        return pars
        
    def species_scaling(self, silent=True):
        """
        This is called for POINT CELLS ONLY
        Adjust all of the conductances and the cell size according to the species requested.
        This scaling should be used ONLY for point models, as no other compartments
        are scaled.
        
        This scaling routine also sets the temperature for the model to a default value. Some models
        can be run at multiple temperatures, and so a default from one of the temperatures is used.
        The calling cell.set_temperature(newtemp) will change the conductances and reinitialize
        the cell to the new temperature settings.
        
        get_cellpars must be called before this is called.
        
        Parameters
        ----------
        
        silent : boolean (default: True)
            run silently (True) or verbosely (False)
        
        """
        assert self.scaled is False  # block double scaling!
        self.scaled = True
         
        soma = self.soma
        if self.status['species'] == 'mouse':
            if self.debug:
                print (f"  Setting conductances for mouse {self.celltype.title():s} {self.status['modelType']:s} cell, Rothman and Manis, 2003")

            self.c_m = 0.9  # default in units of F/cm^2
            self.vrange = [-75., -55.]
            self.i_test_range={'pulse': (-0.15, 0.15, 0.01)}
            self._valid_temperatures = (22., 38.)
            if self.status['temperature'] is None:
                self.set_temperature(22.)

            sf = 1.0
            if self.status['temperature'] == 38.:  # adjust for 2003 model conductance levels at 38
                sf = 3.03  # Q10 of 2, 22->38C. (p3106, R&M2003c)
                # note that kinetics are scaled in the mod file.
            self.set_soma_size_from_Cm(self.pars.cap)
            self.adjust_na_chans(soma, sf=sf)
            soma().kht.gbar = nstomho(self.pars.kht_gbar, self.somaarea)
            soma().klt.gbar = nstomho(self.pars.klt_gbar, self.somaarea)
            soma().ihvcn.gbar = nstomho(self.pars.ih_gbar, self.somaarea)
            soma().leak.gbar = nstomho(self.pars.leak_gbar, self.somaarea)
            self.axonsf = 0.5
            
        else:
            raise ValueError(f"Species {self.status['species']:s} or species-type {self.status['modelType']:s} is not recognized for IC cells")

        self.check_temperature()

    def get_distancemap(self):
        return {'dend': {'klt': {'gradient': 'linear', 'gminf': 0., 'lambda': 200.},
                         'kht': {'gradient': 'llinear', 'gminf': 0., 'lambda': 200.},
                         'nav11': {'gradient': 'linear', 'gminf': 0., 'lambda': 200.}}, # linear with distance, gminf (factor) is multiplied by gbar
                'apic': {'klt': {'gradient': 'linear', 'gminf': 0., 'lambda': 100.},
                         'kht': {'gradient': 'linear', 'gminf': 0., 'lambda': 100.},
                         'nav11': {'gradient': 'exp', 'gminf': 0., 'lambda': 200.}}, # gradients are: flat, linear, exponential
                }

    def add_axon(self):
        Cell.add_axon(self, self.soma, self.somaarea, self.c_m, self.R_a, self.axonsf)

    def add_dendrites(self):
        """
        Add simple unbranched dendrites to basic Rothman Type I models.
        The dendrites have some kht and ih current
        """
        cs = False  # not implemented outside here - internal Cesium.
        nDend = range(4) # these will be simple, unbranced, N=4 dendrites
        dendrites=[]
        for i in nDend:
            dendrites.append(h.Section(cell=self.soma))
        for i in nDend:
            dendrites[i].connect(self.soma)
            dendrites[i].L = 200 # length of the dendrite (not tapered)
            dendrites[i].diam = 1.5 # dendrite diameter
            dendrites[i].nseg = 21 # # segments in dendrites
            dendrites[i].Ra = 150 # ohm.cm
            dendrites[i].insert('kht')
            if cs is False:
                dendrites[i]().kht.gbar = 0.005 # a little Ht
            else:
                dendrites[i]().kht.gbar = 0.0
            dendrites[i].insert('leak') # leak
            dendrites[i]().leak.gbar = 0.0001
            dendrites[i].insert('ihvcn') # some H current
            dendrites[i]().ihvcn.gbar = 0.# 0.001
            dendrites[i]().ihvcn.eh = -43.0
        self.maindend = dendrites
        self.status['dendrites'] = True
        self.add_section(self.maindend, 'maindend')


