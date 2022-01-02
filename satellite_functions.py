

def satellitecentre(pos,N_min,alpha,reps_max=1e3):
    ##################################################
    ##Find the center of the satellite (highest density peak)
    ##by itereatively decreasing sphere around particles and finding the
    ##center of mass within the sphere
    #########
    ##INPUT##
    #########
    #pos: numpy array, Nx3... contains positions for each particle, x,y,z
    #N_min: minimum number of particles within sphere before exiting iteration
    #alpha: factor to decrease sphere size by... at each iteration R = alpha*R
    #reps_max: maximum number of iterations before exiting... in case of infinite while loop
    #########
    ##OUTPUT
    #########
    #COM: numpy array of length 3... gives center of satellite halo
    ##################################################

    N = data.shape[0]

    #First estimate: center of mass
    COM = np.array([np.mean(pos[:,0]), np.mean(pos[:,1]), np.mean(pos[:,2])])
    p = pos-COM
    r = sqrt(pos[:,0]*pos[:,0] + pos[:,1]*pos[:,1] + pos[:,2]*pos[:,2])
    R = max(r) #size of sphere around particles
    N = data.shape[0]

    #Itereatively find center
    reps = 0
    while (N>N_min):
        COM += [np.mean(p[:,0]), np.mean(p[:,1]), np.mean(p[:,2])]
        R = alpha*R #decrease size of sphere
        p = pos-COM
        r = sqrt(p[:,0]*p[:,0] + p[:,1]*p[:,1] + p[:,2]*p[:,2])
        p = p[r<R]
        N  = p.shape[0]
        reps+=1
        if reps==reps_max:
            print("clustercentre didn't converge")
            break

    return COM


def remove_unbound(data,G,m,r_max=np.inf,P0=0.0):
    ##################################################
    ##Removes particles with binding energy less than P0
    ##Make sure data is in the frame of the satellite
    ##Assumes spherical potential
    #########
    ##INPUT##
    #########
    #data: numpy array, Nx7... contains particle ID,x,y,z,vx,vy,vz
    #G: Gravitational constant
    #m: mass of each particle
    #r_max: only include particles within this radius (default infinity)
    #P0: minimum energy of bound particles (default 0)
    #########
    ##OUTPUT
    #########
    #data_bound: N_boundx7... numpy array containing unbound particles
    ##################################################
    data_bound = 1.0*data
    nnew = data.shape[0]
    nold = nnew + 1

    while nold>nnew:
        #print(nold,nnew)
        r = sqrt(data[:,1]**2 +  data[:,2]**2 +  data[:,3]**2)
        data_bound = data_bound[r<r_max]
        E = energies_spherical(data_bound,G,m,P0)
        data_bound = data_bound[E>0]
        nold = int(nnew)
        nnew = data_bound.shape[0]

    return data_bound





def potential_spherical(r,G,m):
    #########################################################
    ##Calculates the potential of an N-body system, assuming its's spherical
    #########################################################
    #########
    ##INPUT##
    #########
    #r: numpy array of length N... radial distance of each particle
    #G: gravitational constant
    #m: mass of each particle
    #########
    ##OUTPUT##
    #########
    #P: potential at position of each particle
    #########################################################
    r[r==0]+=1e-6 #make sure no zeros.

    #Inside Potential
    R_sort, R_ind = np.unique(r, return_inverse=True) #sorted list without repeats, index
    par_int = (np.cumsum(np.concatenate(([0], np.bincount(R_ind)))))[R_ind] #number of interior particles
    P = par_int/r
    #Outside Potential
    counter = Counter(r) # find repeated values
    vals = counter.values() #number of repeats of value in keys
    keys = counter.keys()
    vals = np.array(list(vals), dtype=float) #Convert to arrays
    keys = np.array(list(keys), dtype=float)
    inds = keys.argsort() # increasing order of keys
    reps = vals[inds] #get number of repeats in increasing order
    R_out = reps[::-1] * 1/R_sort[::-1]
    R_out = np.cumsum(R_out) - R_out #sum 1/R for all exterior particles
    R_out = R_out[::-1] #flip direction
    P += R_out[R_ind]

    P = -G*m*P
    return P



def energies_spherical(data,G,m,P=0.0,P0=0.0):
    ##################################################
    ##Calculate the energy of each particle
    ##assumes spherical potential if P not specified
    #########
    ##INPUT##
    #########
    #data: numpy array, Nx7... contains particle ID,x,y,z,vx,vy,vz
    #G: Gravitational constant
    #m: mass of each particle
    #P: can pass the potential of each particle if it is precalculated...
    #   otherwise it will calculate the potential assuming sphericity
    #P0: in case the energy should be shifted by a constant P
    #########
    ##INPUT##
    #########
    #E: Energy of each each particle... energy is defined as in Binney and Tremaine as
    #   E=-(P+K)... negative energies mean the particle is unbound
    ##################################################
    r = sqrt(data[:,1]**2 +  data[:,2]**2 +  data[:,3]**2)
    v = sqrt(data[:,4]**2 +  data[:,5]**2 +  data[:,6]**2)

    if isinstance(P,float) or isinstance(P,int):
        P = potential_spherical(r,G,m)
    E=(-P-P0) - v*v/2.0

    return E
