"""
Written by jh abel, first begun on 24-11-2014.

Collocation module for parameter identification.

"""

# Import modules
import numpy as np
import casadi as cs

# Class for performing collocation analysis
class Collocation:
    """
    Will solve for parameters for a given model, minimizing distance
    between model solution and experimental time-series data.
    """

    def __init__(self, model=None):

        # Set-up the collocation options


        # Set up solver options
        

        # Add the model to this object
        if model:
            AddModel(model)


        def AttachModel(self,model):
            """
            Attaches a CasADi model to this object, sets the model up for
            its eventual evaluation.
            Call like: AttachModel(model)
            """
            self.model = model()
            self.model.init()
            self.EqCount = self.model.input(cs.DAE_X).size()
            self.ParamCount  = self.model.input(cs.DAE_P).size()
            
            # Some of this syntax is borrowed from Peter St. John.
            self.ylabels = [self.model.inputSX(cs.DAE_X)[i].getDescription()
                            for i in xrange(self.NEQ)]
            self.plabels = [self.model.inputSX(cs.DAE_P)[i].getDescription()
                            for i in xrange(self.NP)]
            
            self.pdict = {}
            self.ydict = {}
            
            for par,ind in zip(self.plabels,range(0,self.NP)):
                self.pdict[par] = ind
                
            for par,ind in zip(self.ylabels,range(0,self.NEQ)):
                self.ydict[par] = ind

            # Model now attached to object along with its contents.

if __name__ == "__main__":

    from jha_CommonFiles import CircadianToolbox
    from jha_CommonFiles.Models.tyson2statemodel import model, param, y0n


