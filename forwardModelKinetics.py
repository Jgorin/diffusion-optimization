import math
import numpy as np
import pandas as pd
import torch
import math as math
from jax import numpy as jnp

def forwardModelKinetics(kinetics,lookup_table,tsec,TC): 
    # kinetics: (Ea, lnd0aa_x, fracs_x). To make this compatible with other functions, if there are x fracs, input x-1 fractions, and the code will determine the
    # final fraction.

    R = 0.008314 #gas constant
    #pi = math.acos(jnp.zeros(1)) * 2

    # Infer the number of domains from input
    if len(kinetics) <= 3:
        ndom = 1
    else:
        ndom = (len(kinetics))//2

    # Make a subset of X, removing the Ea so that we have an even number of elements
    temp = kinetics[1:]


    lnD0aa = jnp.tile(temp[0:ndom],(len(TC),1)) # Do this for LnD0aa
    fracstemp = temp[ndom:] # Grab fracs that were input (one will be missing because it is pre-determined by the others)
    fracs = jnp.tile(jnp.concatenate((fracstemp,1-jnp.sum(fracstemp,axis=0,keepdims=True)),axis=-1),(len(TC),1)) # Add the last frac as 1-sum(other fracs)
    Ea = jnp.tile(kinetics[0],(len(TC),ndom)) # Do for Ea



    # Put time and cumulative time in the correct shape
    if ndom > 1:
        tsec = jnp.tile(jnp.reshape(tsec,(-1,1)),(1,Ea.shape[1])) # This is a complicated-looking way of getting tsec into a numdom x numstep matrix for multiplication
        cumtsec = jnp.tile(jnp.reshape(jnp.cumsum(tsec[:,1],axis=0),(-1,1)),(1,Ea.shape[1])) # Same as above, but for cumtsec 

        # Convert TC to TK and put in correct shape for quick computation                                                 
        TK = jnp.tile(jnp.reshape((TC + 273.15),(-1,1)),(1,Ea.shape[1]))


    else:
        cumtsec = jnp.reshape(jnp.cumsum(tsec,-1),(-1,1))
        TK = jnp.reshape(TC+273.15,(-1,1))
        tsec = jnp.reshape(tsec,(-1,1))

    # Calculate D/a^2 for each domain
    Daa = jnp.exp(lnD0aa)*jnp.exp(-Ea/(R*TK))

    # Pre-allocate fraction and Dtaa
    f = jnp.zeros(Daa.shape)
    Dtaa = jnp.zeros(Daa.shape)
    DtaaForSum = jnp.zeros(Daa.shape)
    

    # Calculate Dtaa in incremental (not cumulative) form including the added heating steps

    DtaaForSum = (Daa[0,:]*tsec[0,:]).reshape(1,ndom)
    DtaaForSum = jnp.vstack([DtaaForSum,Daa[1:,:]*(cumtsec[1:,:]-cumtsec[0:-1,:])])
    #DtaaForSum[1:,:] = Daa[1:,:]*(cumtsec[1:,:]-cumtsec[0:-1,:])

    # # I NEED TO ADJUST THIS FOR JAX STILL! IT'S FINE TO HAVE IT OFF FOR TESTING SINCE IT'S A SMALL CORRECTION.
    # for i in range(len(DtaaForSum[0,:])): #This is a really short loop... range of i is # domains. Maybe we could vectorize to improve performance?
    #     if DtaaForSum[0,i] <= 1.347419e-17:
    #         temp = DtaaForSum[0,:]

    #         DtaaForSum[0,i] *= 0
    #     elif DtaaForSum[0,i] >= 4.698221e-06:
    #         pass
    #     else:
    #         DtaaForSum[0,i] *= lookup_table(DtaaForSum[0,i])

    # Calculate Dtaa in cumulative form.
    Dtaa = jnp.cumsum(DtaaForSum, axis = 0)

    
    # Calculate f at each step
    Bt = Dtaa*math.pi**2

    Dtaa = jnp.cumsum(DtaaForSum, axis = 0)

    second_split = Bt>0.0091
    third_split = Bt > 1.8



    #kps_camera = kps_world - jnp.where(selected_rows[:,None], self.pos, 0) #jnp.where 
    

    f = (6/(jnp.pi**(3/2)))*jnp.sqrt((jnp.pi**2)*Dtaa)

    f = jnp.where(second_split,
            (6/(jnp.pi**(3/2)))*jnp.sqrt((jnp.pi**2)*Dtaa)-(3/(jnp.pi**2))*((jnp.pi**2)*Dtaa), 
            f)
    f = jnp.where(third_split,
            1 - (6/(jnp.pi**2))*jnp.exp(-(jnp.pi**2)*Dtaa),
            f)


    

    # Multiply each gas realease by the percent gas located in each domain (prescribed by input)
    f_MDD = f*fracs

    # Renormalize everything by first calculating the fractional releases at each step, summing back up, 
    # and then dividing by the max released in each fraction. This simulates how we would have measured and calculated this in the lab.
    sumf_MDD = jnp.sum(f_MDD,axis=1)

    # If the second heating step gets gas release all the way to 100%, then the rest of the calculation is not necessary. 
    # Return that sumf_MDD == 0
    if (jnp.round(sumf_MDD[2],decimals=6) == 1):
        return jnp.zeros(len(sumf_MDD))
        

    # Remove the two steps we added, recalculate the total sum, and renormalize.

   
    newf = sumf_MDD[0]
    newf = jnp.append(newf, sumf_MDD[1:]-sumf_MDD[0:-1])
    newf = newf[2:]
    normalization_factor = jnp.max(jnp.cumsum(newf,0))
    diffFi= newf/normalization_factor 

    # use equations 5a through c from Fechtig and Kalbitzer for spherical geometry
    # Fechtig and Kalbitzer Equation 5a, for cumulative gas fractions up to 10%
    # special case when i = 1; need to insert 0 for previous amount released


    
    # Resum the gas fractions into cumulative space that doesn't include the two added steps
    sumf_MDD = jnp.cumsum(diffFi,axis=0)

    return sumf_MDD
