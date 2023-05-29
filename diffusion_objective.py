
from forwardModelKinetics import forwardModelKinetics
from jax import numpy as jnp
from dataset import Dataset
import math as math 
import pickle
import jax


class DiffusionObjective():
    def __init__(self, data:Dataset, time_add: jnp.array, temp_add: jnp.array,pickle_path="../lookup_table.pkl",omitValueIndices = []):
        self.dataset = data
        self.lookup_table = pickle.load(open(pickle_path,'rb'))
        self.time_add = time_add
        self.temp_add = temp_add
        #self.omitValueIndices = jnp.array(omitValueIndices)
        
        
        if self.dataset._thr[0] >15:
            time = self.dataset._thr*60 #time in seconds
        else:
            time = self.dataset._thr*3600

        self.tsec = jnp.concatenate([time_add,time])
        self._TC = jnp.concatenate([temp_add,self.dataset._TC])
        self.omitValueIndices = jnp.isin(jnp.arange(len(self.dataset)), jnp.array(omitValueIndices)).astype(int)
        self.grad = jax.grad(self.objective)
    
    
    def __call__(self, X):
        return self.objective(X)
    
    
    def grad(self, X):
        return self.grad(X)

    
    def objective(self, X): #__call__ #evaluate

        data = self.dataset



        # This function calculates the fraction of gas released from each domain
        # in an MDD model during the heating schedule used in the diffusion
        # experiment. Then the fractions released from each domain are combined in
        # proportion to one another as specified by the MDD model, and the
        # diffusivity of each step is calculated. A residual is calculated as the
        # sum of absolute differences between the observed and modeled release
        # fractions over all steps.

        # JOSH, DO I GET TO ASSUME THAT X IS GOING TO COME IN AS A JAX NP ARRAY?
        # X = jnp.array(X)
        total_moles = X[0]
        X = X[1:]
        
        if len(X) <= 3:
            ndom = 1
        else:
            ndom = (len(X))//2

        # Grab the other parameters from the input
        temp = X[1:]

        # Forward model the results so that we can calculate the misfit.
        
        Fi_MDD = forwardModelKinetics(X,self.lookup_table,self.tsec,self._TC) # Gas fraction released for each heating step in model experiment



        Fi_exp = data.Fi_exp #Gas fraction released for each heating step in experiment
   
       
        # Calculate the Fraction released for each heating step in the real experiment
        TrueFracFi = Fi_exp[1:] - Fi_exp[:-1]
        TrueFracFi = jnp.concatenate((jnp.expand_dims(Fi_exp[0], axis=-1), TrueFracFi), axis=-1)
   

        # Calculate the fraction released for each heating step in the modeled experiment
        trueFracMDD = Fi_MDD[1:] - Fi_MDD[:-1]
        trueFracMDD = jnp.concatenate((jnp.expand_dims(Fi_MDD[0], axis=-1), trueFracMDD), axis=-1)
                # Sometimes the forward model predicts kinetics such that ALL the gas would have leaked out during the irradiation and lab storage.
        # In this case, we end up with trueFracMDD == 0, so we should return a high misfit because we know this is not true, else we wouldn't
        # have measured any He in the lab. 
        if jnp.sum(trueFracMDD) == 0:

            return 10.0**8

        exp_moles = data._M
        #total_moles = torch.sum(exp_moles)

        moles_MDD = trueFracMDD * total_moles

        misfit = ((exp_moles-moles_MDD)**2)/(data.uncert**2)
        

        misfit = jnp.where(self.omitValueIndices,jnp.zeros(misfit.shape),misfit)

        #misfit[omitValueIndices] = 0

        # Return the sum of the residuals
        #misfit = torch.sum(misfit)+not_released_flag
        if jnp.isnan(moles_MDD).all():

            return 10.0**8


        #return torch.log(torch.sum(misfit)).item()
        return jnp.sum(misfit)
