#################################################
#                                               #
#                 simDoSe                       #
#     [Sim]ulate [Do]minance and [Se]lection    #
#        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~           #
#         A Wright-Fisher simulator             #
#        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~           #
#                                               #
#          ----- Version 2.7.1 -----            #
#                                               #
#    Daniel J. Balick                           #
#    Written: May 26, 2017                      #
#    Last updated: Nov 10, 2021                 #
#                                               #
#################################################

from __future__ import division, print_function
simdose_version="2.7.1"


__VERSION__ = 20211110

import numpy as np
import scipy as sp
import scipy.stats as spstats
import sys, os
import optparse, itertools
from numpy.random import poisson
from numpy.random import binomial
from numpy.random import shuffle
from numpy.random import choice
import pandas as pd

import time
start_time = time.time()


#### DAN'S GLOBAL OPTIONS
vectorize=0

fixation_loop=0 #slow way to count fixations

bincount_from_numpy = 1 # Use np.bincount() instead of np.histogram()

is_diploid=0

show_plot=0

np.seterr(all='ignore')

random_choice=1

#poisson_genes=1




###################################################################################################

#______________________________________________________________
#______________________________________________________________
#
#                   FUNCTION DEFINITIONS
#______________________________________________________________
#______________________________________________________________

#### this makes sure a directory exists
def assure_path_exists(directory_temp):
    if os.path.exists(directory_temp):
        return
    else:
        os.makedirs(directory_temp)
    return


#______________________________________________________________
#
#           POISSON PDF MANIPULATION FOR RECURRENCE
#______________________________________________________________


def poisson_cutoff(mu_temp,popsize_temp,monomorphic_sites):
    # find cutoff to truncate poisson: below prob 1/L
    min_prob =0.01/monomorphic_sites
    z = np.arange(popsize_temp)+1
    # create a pdf of all nonzero probabilities for lam=popsize*mu for a single site
    pdf_poisson_nonzero = sp.stats.poisson.pmf(z,mu_temp*popsize_temp)
    # find 1+index for max index beyond which probability is negligible
    pcutoff = 1 + max(np.arange(pdf_poisson_nonzero.size)[pdf_poisson_nonzero>min_prob])

    return pcutoff


def truncated_poisson(mu_temp,popsize_temp,pcutoff_temp,foutput):
    # truncate poisson above cutoff
    z = np.arange(pcutoff_temp)
    pdf_poisson_withzero = sp.stats.poisson.pmf(z,mu_temp*popsize_temp)
    # normalize probability
    norm_pois = np.sum(pdf_poisson_withzero)
    # add remaining probability to zero class
    pdf_poisson_withzero[0] += (1 - norm_pois)
    foutput.write("\nRerun poisson vector for multinomial.  Length Poisson vector ="+str(pdf_poisson_withzero.size))
    foutput.write("\n--- Current runtime: %s minutes ---\n\n" % ((time.time() - start_time)/60))
    
    return pdf_poisson_withzero






#_______________________________________________________________
#
#                   MUTATION STEP
#_______________________________________________________________
def mutation_step(mu_temp, L_temp, popsize_temp, counts_temp, foutput):
    
    counts_temp=counts_temp[counts_temp>0.5]
    counts_temp=counts_temp[counts_temp<popsize_temp]
    
    
    singletons = 1 # new mutations enter as singletons
    
    ### Turn on next line to count segregating mutations
    # current_num_mutations = np.count_nonzero(counts_temp)
    
    num_newmutations = np.random.poisson(mu_temp*L_temp*popsize_temp) # POISSON number of new mutations in the pop

#    num_newmutations = np.poisson(mu_temp*L_temp*popsize_temp) #DELTA FUNCTION number of new mutations in the pop
    new_mutations_temp = np.ones(num_newmutations)
    
#    if singletons == 1:
#        init_counts = 1
#        foutput.write("\n New mutations enter as singletons\n")
#        new_counts_temp = init_counts*new_mutations_temp
#    elif singletons==0:
#        #print "New mutations enter at any frequency"
#        #print "ERROR:  FUNCTION NOT WRITTEN YET"
#        #  NEED TO WRITE THIS FUNCTION
#        # init_counts=np.rand.poisson(1,num_newmutations)
#        #new_counts_temp = init_counts

    ###
    ### TURN THIS ON IF MUTATIONS SHOULD ARRIVE AT FREQUENCIES BESIDES SINGLETONS
    #init_counts=np.rand.poisson(1,num_newmutations)
    if singletons==1:
        init_counts = 1
    #foutput.write("New mutations enter as singletons")
    new_counts_temp = init_counts*new_mutations_temp
    counts_extend = np.append(counts_temp, new_counts_temp)

    if L_temp<counts_extend.size:
        foutput.write("\nWARNING: NUMBER OF SEGREGATING SITES EXCEEDS LENGTH OF GENOME...trimming\n")
        counts_extend=counts_extend[:L_temp]

    counts_extend=counts_extend[counts_extend<popsize_temp]
    counts_extend=counts_extend[counts_extend>0.5]

    return counts_extend




def mutation_step_recurrent(mu_temp, L_temp, popsize_temp, counts_temp,foutput):

    enter_as_singletons=0
    
    # extend counts vector with zeros in monomorphic sites
    if L_temp<counts_temp.size:
        foutput.write("\nWARNING: NUMBER OF SEGREGATING SITES EXCEEDS LENGTH OF GENOME...trimming\n")
        counts_temp=counts_temp[:L_temp]

    # count segregating sites
    seg_sites_temp = counts_temp.size
    #this is a vector of length seg_sites with value 2N
    popsize_array= popsize_temp*np.ones(seg_sites_temp)
    #poisson sample new mutations in non-mutated individuals
    recurrent_counts = np.random.poisson((popsize_array - counts_temp)*mu_temp)
    # add new mutatinos to segregating sites
    counts_with_recurrent = counts_temp+recurrent_counts
    #remove any seg sites that end up with counts > 2N
    counts_with_recurrent=counts_with_recurrent[counts_with_recurrent<popsize_temp]
    # deterministic number of new mutations that come in as singletons with rate 2N mu (L-seg_sites)
    num_newmutations = int(popsize_temp*mu_temp*(L_temp-seg_sites_temp))
    # poisson sample number of new mutations for each location
    if enter_as_singletons==1:
        # override and force new mutations to enter as singletons
        new_mutations_temp = np.ones(num_newmutations)
    else:
        # make a vector of num_new_mutations poisson samples around 1
        new_mutations_temp = np.random.poisson(1,num_newmutations)
    new_counts_temp=new_mutations_temp[new_mutations_temp>0.5]
    # can probably comment this out:
    new_counts_temp=new_mutations_temp[new_mutations_temp<popsize_temp]
    counts_extend = np.append(counts_with_recurrent, new_counts_temp)

    counts_extend=counts_extend[counts_extend>0.5]
    counts_extend=counts_extend[counts_extend<popsize_temp]

    if L_temp<counts_temp.size:
        foutput.write("\nWARNING: NUMBER OF SEGREGATING SITES EXCEEDS LENGTH OF GENOME...trimming\n")
        counts_temp=counts_temp[:L_temp]

    return counts_extend





def mutation_step_recurrent_v2(mu_temp, L_temp, popsize_temp, counts_temp,foutput):
    
    # extend counts vector with zeros in monomorphic sites
    if L_temp<counts_temp.size:
        foutput.write("\nWARNING: NUMBER OF SEGREGATING SITES EXCEEDS LENGTH OF GENOME...trimming\n")
        counts_temp=counts_temp[:L_temp]
    
    # count segregating sites
    seg_sites_temp = counts_temp.size
    num_newmutations = int(popsize_temp*mu_temp*(L_temp-seg_sites_temp))
    empty_counts=np.zeros(num_newmutations)
    counts_extend=np.append(counts_temp,empty_counts)
    popsize_array= popsize_temp*np.ones(counts_extend.size)
    recurrent_counts = np.random.poisson((popsize_array - counts_extend)*mu_temp)
    counts_with_recurrent=counts_extend+recurrent_counts
    counts_with_recurrent=counts_with_recurrent[counts_with_recurrent>0.5]
    counts_with_recurrent=counts_with_recurrent[counts_with_recurrent<popsize_temp]

    if L_temp<counts_with_recurrent.size:
        foutput.write("\nWARNING: NUMBER OF SEGREGATING SITES EXCEEDS LENGTH OF GENOME...trimming\n")
        counts_with_recurrent=counts_with_recurrent[:L_temp]

    return counts_with_recurrent







def mutation_step_recurrent_v3(mu_temp, L_temp, popsize_temp, counts_temp,foutput):
    
    # extend counts vector with zeros in monomorphic sites
    if L_temp<counts_temp.size:
        foutput.write("\nWARNING: NUMBER OF SEGREGATING SITES EXCEEDS LENGTH OF GENOME...trimming\n")
        counts_temp=counts_temp[:L_temp]
    
    # count segregating sites
    seg_sites_temp = counts_temp.size
    monomorphic_sites=np.int((L_temp-seg_sites_temp))
    if monomorphic_sites==0:
        new_mutations=np.array([])
#    elif monomorphic_sites==1:
#        new_mutations=np.array(np.random.poisson(popsize_temp*mu_temp))
    else:
        new_mutations=np.random.poisson(popsize_temp*mu_temp,monomorphic_sites)
        new_mutations=new_mutations[new_mutations>0.5]
    popsize_array= popsize_temp*np.ones(counts_temp.size)
    recurrent_counts = np.random.poisson((popsize_array - counts_temp)*mu_temp)
    counts_with_recurrent=counts_temp+recurrent_counts
    counts_with_recurrent=counts_with_recurrent[counts_with_recurrent>0.5]
    counts_extend=np.append(counts_with_recurrent,new_mutations)
    counts_extend=counts_extend[counts_extend>0.5]
    counts_extend=counts_extend[counts_extend<popsize_temp]

    if L_temp<counts_extend.size:
        foutput.write("\nWARNING: NUMBER OF SEGREGATING SITES EXCEEDS LENGTH OF GENOME...trimming\n")
        counts_extend=counts_with_recurrent[:L_temp]

    return counts_extend




def mutation_step_recurrent_vSLOW(mu_temp, L_temp, popsize_temp, counts_temp,foutput):
    
    # extend counts vector with zeros in monomorphic sites
    if L_temp<counts_temp.size:
        foutput.write("\nWARNING: NUMBER OF SEGREGATING SITES EXCEEDS LENGTH OF GENOME...trimming (before mu)\n")
        counts_temp=counts_temp[:L_temp]
    
    # count segregating sites
    seg_sites_temp = counts_temp.size
    monomorphic_sites=np.int((L_temp-seg_sites_temp))
    if monomorphic_sites==0:
        new_mutations=np.array([])
#    elif monomorphic_sites==1:
#        new_mutations=np.array(np.random.poisson(popsize_temp*mu_temp))
    else:
        new_mutations=np.zeros(monomorphic_sites)
    counts_with_zeros = np.append(counts_temp, new_mutations)
    popsize_array= popsize_temp*np.ones(L_temp)
#    popsize_array= popsize_temp*np.ones(counts_with_zeros.size)
    counts_with_recurrent=counts_with_zeros+np.random.poisson((popsize_array - counts_with_zeros)*mu_temp)
    counts_with_recurrent=counts_with_recurrent[counts_with_recurrent>0.5]
    counts_with_recurrent=counts_with_recurrent[counts_with_recurrent<popsize_temp]

    if L_temp<counts_with_recurrent.size:
        foutput.write("\nWARNING: NUMBER OF SEGREGATING SITES EXCEEDS LENGTH OF GENOME...trimming (after mu)\n")
        counts_with_recurrent=counts_with_recurrent[:L_temp]

    return counts_with_recurrent



def mutation_step_recurrent_vMultinomial(mu_temp, L_temp, popsize_temp, counts_temp,poisson_pdf_temp,generation_temp, foutput):
    
    counts_temp=counts_temp[counts_temp>0.5]
    counts_temp=counts_temp[counts_temp<popsize_temp]

    
    
    # extend counts vector with zeros in monomorphic sites
    if L_temp<counts_temp.size:
        foutput.write("\nWARNING: NUMBER OF SEGREGATING SITES EXCEEDS LENGTH OF GENOME...trimming\n")
        counts_temp=counts_temp[:L_temp]
    
    # count segregating sites
    seg_sites_temp = counts_temp.size
    monomorphic_sites=np.int((L_temp-seg_sites_temp))
    if monomorphic_sites==0:
        new_mutations=np.array([])
    elif monomorphic_sites<seg_sites_temp:
        if generation_temp%(popsize_temp/2)==0:
            foutput.write("\nMutations are saturated...stop multinomial\n")
        new_mutations=np.random.poisson(popsize_temp*mu_temp,monomorphic_sites)
        new_mutations=new_mutations[new_mutations>0.5]
    else:
        # dont include zeros in poisson_pdf
        multinomial_list = np.random.multinomial(monomorphic_sites,poisson_pdf_temp)
        new_mutations=np.zeros(np.sum(multinomial_list),dtype=int)
        multinomial_max_nonzero = max(np.arange(len(multinomial_list))[multinomial_list>0.5])
        k_counter=0
        for k in np.arange(1,multinomial_max_nonzero+1):
            new_mutations[k_counter:multinomial_list[k]] = k
            k_counter+=multinomial_list[k]
        new_mutations=new_mutations[new_mutations>0.5]




    popsize_array= popsize_temp*np.ones(counts_temp.size)
    recurrent_counts = np.random.poisson((popsize_array - counts_temp)*mu_temp)
    counts_with_recurrent=counts_temp+recurrent_counts


    counts_extend=np.append(counts_with_recurrent,new_mutations)
    counts_extend=counts_extend[counts_extend>0.5]
    counts_extend=counts_extend[counts_extend<popsize_temp]

    if L_temp<counts_extend.size:
        foutput.write("\nWARNING: NUMBER OF SEGREGATING SITES EXCEEDS LENGTH OF GENOME...trimming\n")
        counts_extend=counts_with_recurrent[:L_temp]

    return counts_extend








#_______________________________________________________________
#
#                  OLD DRIFT STEP (no complete underdominance)
#_______________________________________________________________

def drift_step_old(s_temp, h_temp, new_popsize_temp, old_popsize_temp, counts_temp, fixed_mutations_temp, foutput, generation_temp):
    
    new_counts=np.array([])
    new_counts_no_zero=np.array([])


    s_het = h_temp*s_temp
    s_hom = s_temp
    if np.abs(s_het) > 1:
        s_het = np.sign(s_het) # sets to +/- 1.0 if het effect is larger than one
    
    p = 1.0*counts_temp/old_popsize_temp
    expected_freq_numerator = (1+s_hom)*p*p + (1+s_het)*p*(1-p)
    expected_freq_denom = (1+s_hom)*(p*p) + (1+s_het)*2*p*(1-p) +(1-p)*(1-p)
    expected_freq = expected_freq_numerator/expected_freq_denom
    
    
    
    if expected_freq.size>1.5 and generation_temp>0:
        # MAKE SURE ALL PROBABILITIES ARE LESS THAN ONE
        if np.max(expected_freq)>1:
            foutput.write( "ERROR: Probability greater than one encountered! this!\n")
            foutput.write( "maximum p (frequency) = "+str(np.max(p))+  "\n")
            foutput.write( "maximum probability = "+str(np.max(expected_freq))+  "\n")
            return

    new_counts = np.array(np.random.binomial(new_popsize_temp, expected_freq))
    
    if new_counts.size > 1.5:  # NOT SURE WHY THIS WILL NOT ITERATE IF LENGTH OF THIS VECTOR IS ONE...
        if fixation_loop ==1:
            #_______________________________________________________________
            #   OLD VERSION: Loop over mutations to count fixation events (SLOW!)
            for j in range(new_counts.size):
                if new_counts[j]>=new_popsize_temp:
                    #foutput.write( "\n A mutation should have fixed here. \n")
                    fixed_mutations_temp+=1
                    ####  RESET FIXED MUTATIONS (NO BACK MUTATIONS) #######
                    new_counts[j]=0
            #### SHORTEN LIST BY REMOVING EXTINCT AND FIXED MUTATIONS #######
            new_counts = new_counts[new_counts != 0]
            #_______________________________________________________________



        elif fixation_loop==0:
        
            
            #_______________________________________________________________
            # NEW VERSION: Comupute fixation in vector form.
            # Turn this on to be much faster

            new_counts_no_zero = new_counts[new_counts > 0.5]  # REMOVE EXTINCT MUTATIONS
            new_counts = new_counts_no_zero[new_counts_no_zero < new_popsize_temp] # REMOVE FIXED MUTATIONS
            fixed_mutations_temp+= (new_counts_no_zero.size - new_counts.size) # COUNT REMOVED FIXED MUTATIONS
            #_______________________________________________________________


    return new_counts, fixed_mutations_temp



#_______________________________________________________________
#
#                    DRIFT STEP WITH UNDERDOMINANCE OPTION
#_______________________________________________________________

def drift_step(s_temp, h_temp, complete_underdominance_flag, new_popsize_temp, old_popsize_temp, counts_temp, fixed_mutations_temp, foutput, generation_temp):
    
    new_counts=np.array([])
    new_counts_no_zero=np.array([])

#    Only alter this if running complete underdominance
    if complete_underdominance_flag:
        s_het = h_temp*s_temp
        s_hom = 0
    else:
        s_het = h_temp*s_temp
        s_hom = s_temp
#    s_het = h_temp*s_temp
#    s_hom = s_temp

    if np.abs(s_het) > 1:
        s_het = np.sign(s_het) # sets to +/- 1.0 if het effect is larger than one
    
    p = 1.0*counts_temp/old_popsize_temp
    expected_freq_numerator = (1+s_hom)*p*p + (1+s_het)*p*(1-p)
    expected_freq_denom = (1+s_hom)*(p*p) + (1+s_het)*2*p*(1-p) +(1-p)*(1-p)
    expected_freq = expected_freq_numerator/expected_freq_denom
    
    
    
    if expected_freq.size>1.5 and generation_temp>0:
        # MAKE SURE ALL PROBABILITIES ARE LESS THAN ONE
        if np.max(expected_freq)>1:
            foutput.write( "ERROR: Probability greater than one encountered! this!\n")
            foutput.write( "maximum p (frequency) = "+str(np.max(p))+  "\n")
            foutput.write( "maximum probability = "+str(np.max(expected_freq))+  "\n")
            return

    new_counts = np.array(np.random.binomial(new_popsize_temp, expected_freq))
    
    if new_counts.size > 1.5:  # NOT SURE WHY THIS WILL NOT ITERATE IF LENGTH OF THIS VECTOR IS ONE...
        if fixation_loop ==1:
            #_______________________________________________________________
            #   OLD VERSION: Loop over mutations to count fixation events (SLOW!)
            for j in range(new_counts.size):
                if new_counts[j]>=new_popsize_temp:
                    #foutput.write( "\n A mutation should have fixed here. \n")
                    fixed_mutations_temp+=1
                    ####  RESET FIXED MUTATIONS (NO BACK MUTATIONS) #######
                    new_counts[j]=0
            #### SHORTEN LIST BY REMOVING EXTINCT AND FIXED MUTATIONS #######
            new_counts = new_counts[new_counts != 0]
            #_______________________________________________________________



        elif fixation_loop==0:
        
        
            #_______________________________________________________________
            # NEW VERSION: Comupute fixation in vector form.
            # Turn this on to be much faster

            new_counts_no_zero = new_counts[new_counts > 0.5]  # REMOVE EXTINCT MUTATIONS
            new_counts = new_counts_no_zero[new_counts_no_zero < new_popsize_temp] # REMOVE FIXED MUTATIONS
            fixed_mutations_temp+= (new_counts_no_zero.size - new_counts.size) # COUNT REMOVED FIXED MUTATIONS
            #_______________________________________________________________


    return new_counts, fixed_mutations_temp








#_______________________________________________________________
#
#                   DEMOGRAPHIC MODELS
#_______________________________________________________________


def demography_equilibrium(old_popsize_temp):

    new_popsize_temp = old_popsize_temp

    return new_popsize_temp, old_popsize_temp




def demography_lineargrowth(old_popsize_temp, growthrate_temp, generation_temp):
    

    if generation_temp > 5420:
        new_popsize_temp = old_popsize_temp + int(growthrate_temp*(generation_temp-5420))
    else:
        new_popsize_temp=old_popsize_temp
    
    return new_popsize_temp


def demography_exponential(old_popsize_temp, growthrate_temp, generation_temp):
    
    if generation_temp > 5420:
        new_popsize_temp = int(old_popsize_temp* np.exp(growthrate_temp*(generation_temp-5420)))
    else:
        new_popsize_temp = old_popsize_temp
    
    return new_popsize_temp



def demography_tennessen(initpopsize_temp, generation_temp, ancestry_temp, second_growth_init):
    
    
    if ancestry_temp=="african":
        if generation_temp < 5716:
            new_popsize_temp=initpopsize_temp
            # new_popsize_temp = 28948  ### THIS IS THE DEFAULT ANCESTRAL SIZE
        elif generation_temp>=5716 and generation_temp<=5920:
            new_popsize_temp = int(initpopsize_temp*np.exp(0.0166*(generation_temp-5716)))

    if ancestry_temp=="european":
        if generation_temp < 3880:
            new_popsize_temp=initpopsize_temp
            # new_popsize_temp=28948  ## THIS IS THE DEFAULT ANCESTRAL SIZE
        elif generation_temp>=3880 and generation_temp<5000:
            new_popsize_temp=3722
        elif generation_temp>=5000 and generation_temp<5716:
            new_popsize_temp=int(2064*np.exp(0.00307*(generation_temp-5000)))
            
        elif generation_temp>=5716:
                new_popsize_temp=int(second_growth_init*np.exp(0.0195*(generation_temp-5716)))


        return new_popsize_temp


def demography_super_tennessen(initpopsize_temp, generation_temp, ancestry_temp, second_growth_init, second_growth_rate):
    
    
    if ancestry_temp=="african":
        if generation_temp < 5716:
            new_popsize_temp=initpopsize_temp
            # new_popsize_temp = 28948  ### THIS IS THE DEFAULT ANCESTRAL SIZE
        elif generation_temp>=5716 and generation_temp<=5920:
            new_popsize_temp = int(initpopsize_temp*np.exp(0.0166*(generation_temp-5716)))

    if ancestry_temp=="european":
        if generation_temp < 3880:
            new_popsize_temp=initpopsize_temp
            # new_popsize_temp=28948  ## THIS IS THE DEFAULT ANCESTRAL SIZE
        elif generation_temp>=3880 and generation_temp<5000:
            new_popsize_temp=3722
        elif generation_temp>=5000 and generation_temp<5716:
            new_popsize_temp=int(2064*np.exp(0.00307*(generation_temp-5000)))
            
    if generation_temp>=5716:
            new_popsize_temp=int(second_growth_init*np.exp(second_growth_rate*(generation_temp-5716)))


    return new_popsize_temp


def demography_gaussian_tennessen(initpopsize_temp, generation_temp, ancestry_temp, second_growth_init, second_growth_rate):
    
    
    if ancestry_temp=="african":
        if generation_temp < 5716:
            new_popsize_temp=initpopsize_temp
            # new_popsize_temp = 28948  ### THIS IS THE DEFAULT ANCESTRAL SIZE
        elif generation_temp>=5716 and generation_temp<=5920:
            new_popsize_temp = int(initpopsize_temp*np.exp(0.0166*(generation_temp-5716)))

    if ancestry_temp=="european":
        if generation_temp < 3880:
            new_popsize_temp=initpopsize_temp
            # new_popsize_temp=28948  ## THIS IS THE DEFAULT ANCESTRAL SIZE
        elif generation_temp>=3880 and generation_temp<5000:
            new_popsize_temp=3722
        elif generation_temp>=5000 and generation_temp<5716:
            new_popsize_temp=int(2064*np.exp(0.00307*(generation_temp-5000)))
            
    if generation_temp>=5716:
            new_popsize_temp=int(second_growth_init*np.exp(second_growth_rate*(generation_temp-5716)*(generation_temp-5716)))


    return new_popsize_temp


def demography_browning(initpopsize_temp, generation_temp, ancestry_temp, second_growth_init, third_growth_init, third_growth_rate):
    
    
    if ancestry_temp=="african":
        if generation_temp < 5716:
            new_popsize_temp=initpopsize_temp
            # new_popsize_temp = 28948  ### THIS IS THE DEFAULT ANCESTRAL SIZE
        elif generation_temp>=5716 and generation_temp<=5920:
            new_popsize_temp = int(initpopsize_temp*np.exp(0.0166*(generation_temp-5716)))

    if ancestry_temp=="european":
        if generation_temp < 3880:
            new_popsize_temp=initpopsize_temp
            # new_popsize_temp=28948  ## THIS IS THE DEFAULT ANCESTRAL SIZE
        elif generation_temp>=3880 and generation_temp<5000:
            new_popsize_temp=3722
        elif generation_temp>=5000 and generation_temp<5716:
            new_popsize_temp=int(2064*np.exp(0.00307*(generation_temp-5000)))
            
    if generation_temp>=5716 and generation_temp < 5903:
            new_popsize_temp=int(second_growth_init*np.exp(0.0195*(generation_temp-5716)))

    if generation_temp>=5903:
            new_popsize_temp=int(third_growth_init*10**(third_growth_rate*(generation_temp-5903)))

    return new_popsize_temp





#_______________________________________________________________
#
#   OLD DEMOGRAPHIC FUNCTIONS FROM DR+DMJ PYTHON CODE
#_______________________________________________________________
#def newgravel(gen, pop, speed, size):
#    pop[0] = 28948
#    if gen<3880:
#        pop[1]=pop[0]
#    elif gen>=3880 and gen<5000:
#        pop[1]=3722
#    elif gen>=5000:
#        pop[1]= np.floor(2064*np.exp(0.003857868074807*(gen-5000)))
#    # end time is 5920
#    pop[:] = np.floor(0.5 + pop/speed)
#
#def tennessen(gen, pop, speed, size):
#    if gen<5716:
#        pop[0]=28948
#    elif gen>=5716 and gen<=5920:
#        pop[0]=np.floor(28948*np.exp(0.0166*(gen-5716)))
#    if gen<3880:
#        pop[1]=28948
#    elif gen>=3880 and gen<5000:
#        pop[1]=3722
#    elif gen>=5000 and gen<5716:
#        pop[1]=np.floor(2064*np.exp(0.00307*(gen-5000)))
#    elif gen>=5716:
#        pop[1]=np.floor(18900*np.exp(0.0195*(gen-5716)))
#    # end time is 5920
#    pop[:] = np.floor(0.5 + pop/speed)
#
#def simplebottle(gen, pop, speed, size):
#    pop[0]=pop[1]=28948;
#        if gen>=3880 and gen<4000:
#            pop[1]=290
#    pop[:] = np.floor(0.5 + pop/speed)



#
##_______________________________________________________________
##
##                   POPULATION DOWNSAMPLING
##_______________________________________________________________
#
###################
##  IGNORE THESE FUNCTIONS FOR NOW.  THEY ARE REPLACED WITH A SINGLE GENERATION OF DRIFT
####################
#
#### THIS ONE IS SLOWEST AND DOES NOT WORK
#def downsample_SFS(popsize_temp, SFS_temp, sample_size):
#    
#    SFS_frequency = SFS_temp/popsize_temp # original SFS in frequency space
#    sample_counts = 1+np.arange(sample_size-1) # only consider segregating sites (no k=0 or k=2m counts)
#    sample_SFS_temp = np.zeros(sample_counts.size)
#    for i in range(SFS_frequency.size):
#        for j in range(sample_counts.size):
#            sample_SFS_temp[j]+=np.random.poisson(sample_counts[j]*SFS_frequency[i])
#    return sample_SFS_temp
#
#
#### THIS ONE IS FASTEST AND DOES NOT WORK
#def downsample_SFS_one_loop(popsize_temp, SFS_temp, sample_size):
#    
#    SFS_frequency = SFS_temp/popsize_temp # original SFS in frequency space
#    sample_counts = 1+np.arange(sample_size-1) # only consider segregating sites (no k=0 or k=2m counts)
#    sample_SFS_temp = np.zeros(sample_counts.size)
#    temp_array=np.zeros(SFS_frequency.size)
#    for j in range(sample_counts.size):
#            temp_array=np.random.poisson(sample_counts[j]*SFS_frequency)
#            sample_SFS_temp[j]=np.sum(temp_array)
#    return sample_SFS_temp
#
#
#### THIS ONE IS INTERMEDIATE SPEED (HAS TO MAKE GIANT MATRIX) AND DOES NOT WORK
#def downsample_SFS_no_loops(popsize_temp, SFS_temp, sample_size):
#    
#    SFS_frequency = SFS_temp/popsize_temp # original SFS in frequency space
#    sample_counts = 1+np.arange(sample_size-1) # only consider segregating sites (no k=0 or k=2m counts)
#    old_SFS_matrix = np.outer(SFS_frequency,sample_counts)
#    new_sample_matrix = np.random.poisson(old_SFS_matrix)
#    sample_SFS_temp = new_sample_matrix.sum(0)
#    
#    return sample_SFS_temp
#



#_______________________________________________________________
#
#                  SIMULATE GENES OF FIXED LENGTH
#_______________________________________________________________


def  simulate_genes(number_genes_temp, gene_length_temp, mutation_rate_temp,sample_counts_temp, genome_length, sample_size_temp, rare_number_temp, foutput):
    # Remove any fixed mutations if they still exist
    sample_counts_temp = sample_counts_temp[sample_counts_temp<sample_size_temp]
    sample_counts_temp = sample_counts_temp[sample_counts_temp>0.5]



    monomorphic_sites = genome_length-sample_counts_temp.size
    add_zeros = np.zeros(monomorphic_sites)
    all_sites = np.append(sample_counts_temp, add_zeros)
    all_sites_int = all_sites.astype(int)

    total_counts_gene = np.zeros(number_genes_temp)
    xbar_gene = np.zeros(number_genes_temp)
    x2bar_gene = np.zeros(number_genes_temp)
    pi_gene = np.zeros(number_genes_temp)
    logstat_gene = np.zeros(number_genes_temp)
    logstatUscale_gene = np.zeros(number_genes_temp)
    logstatMUscale_gene = np.zeros(number_genes_temp)
    logstatLscale_gene = np.zeros(number_genes_temp)
    logload_gene = np.zeros(number_genes_temp)
    logload2_gene = np.zeros(number_genes_temp)
    segregating_sites_gene = np.zeros(number_genes_temp)
    singletons_gene = np.zeros(number_genes_temp)
    frac_rare_gene = np.zeros(number_genes_temp)
    
    geneSFS=[]
    
    for gene in range(number_genes_temp):
        
        MUgene=mutation_rate_temp
        Ugene =mutation_rate_temp*gene_length_temp
        Lgene = gene_length_temp
        
        if gene%5000==0:
            foutput.write("\n--- Current runtime: %s minutes ---\n" % ((time.time() - start_time)/60))
            foutput.write("\nDone simulating "+str(gene)+" genes\n")
        
        if random_choice==1:
            gene_sites = np.random.choice(all_sites_int,gene_length_temp)
        else:
            np.random.shuffle(all_sites_int)
            gene_sites = all_sites_int[:gene_length_temp] # this gives an array of length gene_length_temp
        
  
        geneSFS.append(list(gene_sites))
                
#
#        ####  COMPUTE MOMENTS USING FREQUENCY
#        gene_freq = gene_sites/sample_size_temp
#        xbar_gene[gene] = np.sum(gene_freq) # sum over frequencies
#        x2bar_gene[gene] = np.sum(gene_freq*gene_freq) # sum over squared frequency
#        

        # compute xbar
        total_counts_gene[gene]=np.sum(gene_sites)
        xbar_gene[gene] = np.sum(gene_sites)/sample_size_temp # sum over counts
        # compute x2bar
        x2bar_gene[gene] = np.sum(gene_sites*gene_sites)/(sample_size_temp*sample_size_temp) # sum over squared frequency
        #compute pi
        pi_gene[gene] = 2*(xbar_gene[gene]-x2bar_gene[gene])
        # make gene SFS
        gene_SFS = np.bincount(gene_sites, minlength=(sample_size_temp+1))
        #remove zero counts and fixed counts
        gene_SFS = gene_SFS[1:sample_size_temp] # note [:sample_size+1] includes fixed
        # create Log(1+SFS)
        log1plusSFS = np.log1p(gene_SFS)
        # sum to make logstat
        logstat_gene[gene] = np.sum(log1plusSFS)
        # create Log(1+SFS/U)
        log1plusSFSUscale = np.log1p(gene_SFS/Ugene)
        # sum to make logstatUscale
        logstatUscale_gene[gene] = np.sum(log1plusSFSUscale)
        # create Log(1+SFS/U)
        log1plusSFSMUscale = np.log1p(gene_SFS/MUgene)
        # sum to make logstatMUscale
        logstatMUscale_gene[gene] = np.sum(log1plusSFSMUscale)
        # create Log(1+SFS/U)
        log1plusSFSLscale = np.log1p(gene_SFS/Lgene)
        # sum to make logstatLscale
        logstatLscale_gene[gene] = np.sum(log1plusSFSLscale)


        # create frequency
        x_SFS=(1+np.arange(gene_SFS.size))/sample_size_temp
        # sum to make logload = x*log(1+phi)
        logload_gene[gene] = np.sum(x_SFS*log1plusSFS)
        # sum to make logload2 = x^2*log(1+phi)
        logload2_gene[gene] = np.sum(x_SFS*x_SFS*log1plusSFS)
        
        # compute segregating sites
        segregating_sites_gene[gene] = np.sum(gene_SFS)
        # compute singletons
        singletons_gene[gene] = gene_SFS[0]
        # compute fraction of rare
        frac_rare_gene[gene] = np.sum(gene_SFS[:rare_number_temp])/segregating_sites_gene[gene]
    

    

    return np.array(geneSFS), total_counts_gene, xbar_gene, x2bar_gene, pi_gene, logstat_gene,logstatUscale_gene,logstatMUscale_gene,logstatLscale_gene, logload_gene, logload2_gene, segregating_sites_gene, singletons_gene, frac_rare_gene



def  simulate_genes_poisson(number_genes_temp, gene_length_temp, mutation_rate_temp,sample_SFS_temp, genome_length, sample_size_temp, rare_number_temp, foutput):
#    # add monomorphic sites to the SFS as a zero bin
#    monomorphic_sites = genome_length-np.sum(sample_SFS_temp)
#    all_sites_SFS = np.append(np.array([monomorphic_sites]), sample_counts_temp)
#    all_sites_SFS_int = all_sites_SFS.astype(int)

    # make empty lists of gene stats
    total_counts_gene = np.zeros(number_genes_temp)
    xbar_gene = np.zeros(number_genes_temp)
    x2bar_gene = np.zeros(number_genes_temp)
    pi_gene = np.zeros(number_genes_temp)
    logstat_gene = np.zeros(number_genes_temp)
    logstatUscale_gene = np.zeros(number_genes_temp)
    logstatMUscale_gene = np.zeros(number_genes_temp)
    logstatLscale_gene = np.zeros(number_genes_temp)
    logload_gene=np.zeros(number_genes_temp)
    logload2_gene=np.zeros(number_genes_temp)
    segregating_sites_gene = np.zeros(number_genes_temp)
    singletons_gene = np.zeros(number_genes_temp)
    frac_rare_gene = np.zeros(number_genes_temp)
    
#    foutput.write("shape of sample SFS = "+str(np.shape(sample_SFS_temp)))

    for gene in range(number_genes_temp):
        
        MUgene=mutation_rate_temp
        Ugene =mutation_rate_temp*gene_length_temp
        Lgene = gene_length_temp
        
        
        if gene%5000==0:
            foutput.write("\n--- Current runtime: %s minutes ---\n" % ((time.time() - start_time)/60))
            foutput.write("\nDone simulating "+str(gene)+" genes\n")
        
        gene_SFS = np.random.poisson((gene_length_temp/genome_length)*sample_SFS_temp)
#        foutput.write("shape of gene SFS = "+str(np.shape(gene_SFS)))
        # ignore fixed sites/monomorphic sites for now
        gene_counter = (np.arange(sample_size_temp-1)+1)
        gene_freq = (np.arange(sample_size_temp-1)+1)/sample_size_temp
#        foutput.write("shape of frequency array = "+str(np.shape(gene_freq)))
        # compute xbar
        total_counts_gene[gene] = np.sum(gene_counter*gene_SFS)
        xbar_gene[gene] = np.sum(gene_freq*gene_SFS)
        # compute x2bar
        x2bar_gene[gene] = np.sum(gene_freq*gene_freq*gene_SFS)
        #compute pi
        pi_gene[gene] = 2*(xbar_gene[gene]-x2bar_gene[gene])
        # create Log(1+SFS)
        log1plusSFS = np.log1p(gene_SFS)
        # sum to make logstat
        logstat_gene[gene] = np.sum(log1plusSFS)
        # create Log(1+SFS/U)
        log1plusSFSUscale = np.log1p(gene_SFS/Ugene)
        # sum to make logstatUscale
        logstatUscale_gene[gene] = np.sum(log1plusSFSUscale)
        # create Log(1+SFS/U)
        log1plusSFSMUscale = np.log1p(gene_SFS/MUgene)
        # sum to make logstatMUscale
        logstatMUscale_gene[gene] = np.sum(log1plusSFSMUscale)
        # create Log(1+SFS/U)
        log1plusSFSLscale = np.log1p(gene_SFS/Lgene)
        # sum to make logstatLscale
        logstatLscale_gene[gene] = np.sum(log1plusSFSLscale)
        
        # create frequency
        x_SFS=(1+np.arange(gene_SFS.size))/sample_size_temp
        # sum to make logload = x*log(1+phi)
        logload_gene[gene] = np.sum(x_SFS*log1plusSFS)
        # sum to make logload2 = x^2*log(1+phi)
        logload2_gene[gene] = np.sum(x_SFS*x_SFS*log1plusSFS)
        
        # compute segregating sites
        segregating_sites_gene[gene] = np.sum(gene_SFS)
        # compute singletons
        singletons_gene[gene] = gene_SFS[0]
        # compute fraction of rare
        frac_rare_gene[gene] = np.sum(gene_SFS[:rare_number_temp])/segregating_sites_gene[gene]
    
    
    return total_counts_gene, xbar_gene, x2bar_gene, pi_gene, logstat_gene,logstatUscale_gene,logstatMUscale_gene,logstatLscale_gene, logload_gene, logload2_gene, segregating_sites_gene, singletons_gene, frac_rare_gene



#_______________________________________________________________
#
#          SIMULATE GENES SAMPLED FROM A LENGTH DISTRIBUTION
#_______________________________________________________________


def  simulate_genes_fixed_list(length_list_temp, mutation_rate_temp,number_genes_temp, sample_counts_temp, sample_SFS_temp, genome_length, sample_size_temp, poisson_genes_flag, rare_number_temp, foutput):
    

    # Remove any fixed mutations if they still exist
    sample_counts_temp = sample_counts_temp[sample_counts_temp<sample_size_temp]
    sample_counts_temp = sample_counts_temp[sample_counts_temp>0.5]

    
    total_counts_gene = np.zeros(number_genes_temp)
    xbar_gene = np.zeros(number_genes_temp)
    x2bar_gene = np.zeros(number_genes_temp)
    pi_gene = np.zeros(number_genes_temp)
    logstat_gene = np.zeros(number_genes_temp)
    logstatUscale_gene = np.zeros(number_genes_temp)
    logstatMUscale_gene = np.zeros(number_genes_temp)
    logstatLscale_gene = np.zeros(number_genes_temp)
    logload_gene = np.zeros(number_genes_temp)
    logload2_gene = np.zeros(number_genes_temp)
    segregating_sites_gene = np.zeros(number_genes_temp)
    singletons_gene = np.zeros(number_genes_temp)
    frac_rare_gene = np.zeros(number_genes_temp)
    
    for gene in range(length_list_temp.size):
        
        MUgene=mutation_rate_temp
        Ugene =mutation_rate_temp*length_list_temp[gene]
        Lgene = length_list_temp[gene]
        
        
        if gene%5000==0:
            foutput.write("\n--- Current runtime: %s minutes ---\n" % ((time.time() - start_time)/60))
            foutput.write("\nDone simulating "+str(gene)+" genes\n")

        if poisson_genes_flag==1:

            gene_SFS = np.random.poisson((length_list_temp[gene]/genome_length)*sample_SFS_temp)
    #        foutput.write("shape of gene SFS = "+str(np.shape(gene_SFS)))
            # ignore fixed sites/monomorphic sites for now
            gene_counter = (np.arange(sample_size_temp-1)+1)
            gene_freq = (np.arange(sample_size_temp-1)+1)/sample_size_temp
    #        foutput.write("shape of frequency array = "+str(np.shape(gene_freq)))
            # compute xbar
            total_counts_gene[gene] = np.sum(gene_counter*gene_SFS)
            xbar_gene[gene] = np.sum(gene_freq*gene_SFS)
            # compute x2bar
            x2bar_gene[gene] = np.sum(gene_freq*gene_freq*gene_SFS)
            #compute pi
            pi_gene[gene] = 2*(xbar_gene[gene]-x2bar_gene[gene])
            # create Log(1+SFS)
            log1plusSFS = np.log1p(gene_SFS)
            # sum to make logstat
            logstat_gene[gene] = np.sum(log1plusSFS)
            # create Log(1+SFS/U)
            log1plusSFSUscale = np.log1p(gene_SFS/Ugene)
            # sum to make logstatUscale
            logstatUscale_gene[gene] = np.sum(log1plusSFSUscale)
            # create Log(1+SFS/U)
            log1plusSFSMUscale = np.log1p(gene_SFS/MUgene)
            # sum to make logstatMUscale
            logstatMUscale_gene[gene] = np.sum(log1plusSFSMUscale)
            # create Log(1+SFS/U)
            log1plusSFSLscale = np.log1p(gene_SFS/Lgene)
            # sum to make logstatLscale
            logstatLscale_gene[gene] = np.sum(log1plusSFSLscale)
            
            # create frequency
            x_SFS=(1+np.arange(gene_SFS.size))/sample_size_temp
            # sum to make logload = x*log(1+phi)
            logload_gene[gene] = np.sum(x_SFS*log1plusSFS)
            # sum to make logload2 = x^2*log(1+phi)
            logload2_gene[gene] = np.sum(x_SFS*x_SFS*log1plusSFS)
            
            # compute segregating sites
            segregating_sites_gene[gene] = np.sum(gene_SFS)
            # compute singletons
            singletons_gene[gene] = gene_SFS[0]
            # compute fraction of rare
            frac_rare_gene[gene] = np.sum(gene_SFS[:rare_number_temp])/segregating_sites_gene[gene]

        else:
            if genome_length >=sample_counts_temp.size:
                monomorphic_sites = genome_length-sample_counts_temp.size
            else:
                    monomorphic_sites=0
            add_zeros = np.zeros(monomorphic_sites)
            all_sites = np.append(sample_counts_temp, add_zeros)
            all_sites_int = all_sites.astype(int)
            # choose sites from length_list rather than from a fixed length
            gene_sites = np.random.choice(all_sites_int,length_list_temp[gene])
            # compute xbar
            total_counts_gene[gene]=np.sum(gene_sites)
            xbar_gene[gene] = np.sum(gene_sites)/sample_size_temp # sum over counts
            # compute x2bar
            x2bar_gene[gene] = np.sum(gene_sites*gene_sites)/(sample_size_temp*sample_size_temp) # sum over squared frequency
            #compute pi
            pi_gene[gene] = 2*(xbar_gene[gene]-x2bar_gene[gene])
            # make gene SFS
            gene_SFS = np.bincount(gene_sites, minlength=(sample_size_temp+1))
            #remove zero counts and fixed counts
            gene_SFS = gene_SFS[1:sample_size_temp] # note [:sample_size+1] includes fixed
            # create Log(1+SFS)
            log1plusSFS = np.log1p(gene_SFS)
            # sum to make logstat
            logstat_gene[gene] = np.sum(log1plusSFS)
            # create Log(1+SFS/U)
            log1plusSFSUscale = np.log1p(gene_SFS/Ugene)
            # sum to make logstatUscale
            logstatUscale_gene[gene] = np.sum(log1plusSFSUscale)
            # create Log(1+SFS/U)
            log1plusSFSMUscale = np.log1p(gene_SFS/MUgene)
            # sum to make logstatMUscale
            logstatMUscale_gene[gene] = np.sum(log1plusSFSMUscale)
            # create Log(1+SFS/U)
            log1plusSFSLscale = np.log1p(gene_SFS/Lgene)
            # sum to make logstatLscale
            logstatLscale_gene[gene] = np.sum(log1plusSFSLscale)
            
            # create frequency
            x_SFS=(1+np.arange(gene_SFS.size))/sample_size_temp
            # sum to make logload = x*log(1+phi)
            logload_gene[gene] = np.sum(x_SFS*log1plusSFS)
            # sum to make logload2 = x^2*log(1+phi)
            logload2_gene[gene] = np.sum(x_SFS*x_SFS*log1plusSFS)
            
            # compute segregating sites
            segregating_sites_gene[gene] = np.sum(gene_SFS)
            # compute singletons
            singletons_gene[gene] = gene_SFS[0]
            # compute fraction of rare
            frac_rare_gene[gene] = np.sum(gene_SFS[:rare_number_temp])/segregating_sites_gene[gene]

    return total_counts_gene, xbar_gene, x2bar_gene, pi_gene, logstat_gene,logstatUscale_gene,logstatMUscale_gene,logstatLscale_gene, logload_gene,logload2_gene, segregating_sites_gene, singletons_gene, frac_rare_gene







#___________________________________________________________________________________________________
#
#          SIMULATE GENES SAMPLED FROM A LENGTH DISTRIBUTION (and print SFS for each gene)
#___________________________________________________________________________________________________


def  simulate_genes_fixed_list_with_SFSprint(length_list_temp, mutation_rate_temp,number_genes_temp, sample_counts_temp, sample_SFS_temp, genome_length, sample_size_temp, poisson_genes_flag, rare_number_temp, initpopsize_temp, demography_type,ancestry_type, s_temp, h_temp, growth_rate_temp, sparse_temp,linearS,linearL,linearU, foutput):
    
    print_each_gene_SFS_flag_temp=1
    
    # Remove any fixed mutations if they still exist
    sample_counts_temp = sample_counts_temp[sample_counts_temp<sample_size_temp]
    sample_counts_temp = sample_counts_temp[sample_counts_temp>0.5]
    
    
    total_counts_gene = np.zeros(number_genes_temp)
    xbar_gene = np.zeros(number_genes_temp)
    x2bar_gene = np.zeros(number_genes_temp)
    pi_gene = np.zeros(number_genes_temp)
    logstat_gene = np.zeros(number_genes_temp)
    logstatUscale_gene = np.zeros(number_genes_temp)
    logstatMUscale_gene = np.zeros(number_genes_temp)
    logstatLscale_gene = np.zeros(number_genes_temp)
    logload_gene = np.zeros(number_genes_temp)
    logload2_gene = np.zeros(number_genes_temp)
    segregating_sites_gene = np.zeros(number_genes_temp)
    singletons_gene = np.zeros(number_genes_temp)
    frac_rare_gene = np.zeros(number_genes_temp)
    
    for gene in range(length_list_temp.size):
        
        MUgene=mutation_rate_temp
        Ugene =mutation_rate_temp*length_list_temp[gene]
        Lgene = length_list_temp[gene]
        
        
        if gene%5000==0:
            foutput.write("\n--- Current runtime: %s minutes ---\n" % ((time.time() - start_time)/60))
            foutput.write("\nDone simulating "+str(gene)+" genes\n")
        
        if poisson_genes_flag==1:
            
            gene_SFS = np.random.poisson((length_list_temp[gene]/genome_length)*sample_SFS_temp)
            #        foutput.write("shape of gene SFS = "+str(np.shape(gene_SFS)))
            # ignore fixed sites/monomorphic sites for now
            gene_counter = (np.arange(sample_size_temp-1)+1)
            gene_freq = (np.arange(sample_size_temp-1)+1)/sample_size_temp
            #        foutput.write("shape of frequency array = "+str(np.shape(gene_freq)))
            # compute xbar
            total_counts_gene[gene] = np.sum(gene_counter*gene_SFS)
            xbar_gene[gene] = np.sum(gene_freq*gene_SFS)
            # compute x2bar
            x2bar_gene[gene] = np.sum(gene_freq*gene_freq*gene_SFS)
            #compute pi
            pi_gene[gene] = 2*(xbar_gene[gene]-x2bar_gene[gene])
            # create Log(1+SFS)
            log1plusSFS = np.log1p(gene_SFS)
            # sum to make logstat
            logstat_gene[gene] = np.sum(log1plusSFS)
            # create Log(1+SFS/U)
            log1plusSFSUscale = np.log1p(gene_SFS/Ugene)
            # sum to make logstatUscale
            logstatUscale_gene[gene] = np.sum(log1plusSFSUscale)
            # create Log(1+SFS/U)
            log1plusSFSMUscale = np.log1p(gene_SFS/MUgene)
            # sum to make logstatMUscale
            logstatMUscale_gene[gene] = np.sum(log1plusSFSMUscale)
            # create Log(1+SFS/U)
            log1plusSFSLscale = np.log1p(gene_SFS/Lgene)
            # sum to make logstatLscale
            logstatLscale_gene[gene] = np.sum(log1plusSFSLscale)
            
            # create frequency
            x_SFS=(1+np.arange(gene_SFS.size))/sample_size_temp
            # sum to make logload = x*log(1+phi)
            logload_gene[gene] = np.sum(x_SFS*log1plusSFS)
            # sum to make logload2 = x^2*log(1+phi)
            logload2_gene[gene] = np.sum(x_SFS*x_SFS*log1plusSFS)
            
            # compute segregating sites
            segregating_sites_gene[gene] = np.sum(gene_SFS)
            # compute singletons
            singletons_gene[gene] = gene_SFS[0]
            # compute fraction of rare
            frac_rare_gene[gene] = np.sum(gene_SFS[:rare_number_temp])/segregating_sites_gene[gene]
    
        else:
            if genome_length >=sample_counts_temp.size:
                monomorphic_sites = genome_length-sample_counts_temp.size
            else:
                monomorphic_sites=0
            add_zeros = np.zeros(monomorphic_sites)
            all_sites = np.append(sample_counts_temp, add_zeros)
            all_sites_int = all_sites.astype(int)
            # choose sites from length_list rather than from a fixed length
            gene_sites = np.random.choice(all_sites_int,length_list_temp[gene])
            # compute xbar
            total_counts_gene[gene]=np.sum(gene_sites)
            xbar_gene[gene] = np.sum(gene_sites)/sample_size_temp # sum over counts
            # compute x2bar
            x2bar_gene[gene] = np.sum(gene_sites*gene_sites)/(sample_size_temp*sample_size_temp) # sum over squared frequency
            #compute pi
            pi_gene[gene] = 2*(xbar_gene[gene]-x2bar_gene[gene])
            # make gene SFS
            gene_SFS = np.bincount(gene_sites, minlength=(sample_size_temp+1))
            #remove zero counts and fixed counts
            gene_SFS = gene_SFS[1:sample_size_temp] # note [:sample_size+1] includes fixed
            # create Log(1+SFS)
            log1plusSFS = np.log1p(gene_SFS)
            # sum to make logstat
            logstat_gene[gene] = np.sum(log1plusSFS)
            # create Log(1+SFS/U)
            log1plusSFSUscale = np.log1p(gene_SFS/Ugene)
            # sum to make logstatUscale
            logstatUscale_gene[gene] = np.sum(log1plusSFSUscale)
            # create Log(1+SFS/U)
            log1plusSFSMUscale = np.log1p(gene_SFS/MUgene)
            # sum to make logstatMUscale
            logstatMUscale_gene[gene] = np.sum(log1plusSFSMUscale)
            # create Log(1+SFS/U)
            log1plusSFSLscale = np.log1p(gene_SFS/Lgene)
            # sum to make logstatLscale
            logstatLscale_gene[gene] = np.sum(log1plusSFSLscale)
            
            # create frequency
            x_SFS=(1+np.arange(gene_SFS.size))/sample_size_temp
            # sum to make logload = x*log(1+phi)
            logload_gene[gene] = np.sum(x_SFS*log1plusSFS)
            # sum to make logload2 = x^2*log(1+phi)
            logload2_gene[gene] = np.sum(x_SFS*x_SFS*log1plusSFS)
            
            # compute segregating sites
            segregating_sites_gene[gene] = np.sum(gene_SFS)
            # compute singletons
            singletons_gene[gene] = gene_SFS[0]
            # compute fraction of rare
            frac_rare_gene[gene] = np.sum(gene_SFS[:rare_number_temp])/segregating_sites_gene[gene]
        
        if print_each_gene_SFS_flag_temp==1:
            assure_path_exists("each_gene_SFS")
            os.chdir("each_gene_SFS")
            fixed_mutations_temp=0
            file_prefix="singleGeneSFS"
            gene_seed=gene
            print_raw_SFS_flag_temp=1
            hist_instead_of_counts_flag_temp=1
            irrelevant_hist, irrelevant_singleton = print_raw_SFS(initpopsize_temp, sample_size_temp, gene_SFS, fixed_mutations_temp, demography_type,ancestry_type, s_temp, h_temp, mutation_rate_temp,length_list_temp[gene], growth_rate_temp, sparse_temp, linearS,linearL,linearU, file_prefix, sample_size_temp, gene_seed, print_raw_SFS_flag_temp, hist_instead_of_counts_flag_temp)
            os.chdir("..")
    
    return total_counts_gene, xbar_gene, x2bar_gene, pi_gene, logstat_gene,logstatUscale_gene,logstatMUscale_gene,logstatLscale_gene, logload_gene,logload2_gene, segregating_sites_gene, singletons_gene, frac_rare_gene


#___________________________________________________________________________________________________
#
#          SIMULATE POISSON GENES OF FIXED LENGTH(and print SFS for each gene)
#___________________________________________________________________________________________________



def  simulate_genes_poisson_with_SFSprint(number_genes_temp, gene_length_temp, mutation_rate_temp,sample_SFS_temp, genome_length, sample_size_temp, rare_number_temp, initpopsize_temp, demography_type,ancestry_type, s_temp, h_temp, growth_rate_temp, sparse_temp,linearS,linearL,linearU, seedstart, foutput):
    
    print_each_gene_SFS_flag_temp=1
    
    
#    # add monomorphic sites to the SFS as a zero bin
#    monomorphic_sites = genome_length-np.sum(sample_SFS_temp)
#    all_sites_SFS = np.append(np.array([monomorphic_sites]), sample_counts_temp)
#    all_sites_SFS_int = all_sites_SFS.astype(int)

    # make empty lists of gene stats
    total_counts_gene = np.zeros(number_genes_temp)
    xbar_gene = np.zeros(number_genes_temp)
    x2bar_gene = np.zeros(number_genes_temp)
    pi_gene = np.zeros(number_genes_temp)
    logstat_gene = np.zeros(number_genes_temp)
    logstatUscale_gene = np.zeros(number_genes_temp)
    logstatMUscale_gene = np.zeros(number_genes_temp)
    logstatLscale_gene = np.zeros(number_genes_temp)
    logload_gene=np.zeros(number_genes_temp)
    logload2_gene=np.zeros(number_genes_temp)
    segregating_sites_gene = np.zeros(number_genes_temp)
    singletons_gene = np.zeros(number_genes_temp)
    frac_rare_gene = np.zeros(number_genes_temp)
    
#    foutput.write("shape of sample SFS = "+str(np.shape(sample_SFS_temp)))

    for gene in range(number_genes_temp):
        
        MUgene=mutation_rate_temp
        Ugene =mutation_rate_temp*gene_length_temp
        Lgene = gene_length_temp
        
        
        if gene%5000==0:
            foutput.write("\n--- Current runtime: %s minutes ---\n" % ((time.time() - start_time)/60))
            foutput.write("\nDone simulating "+str(gene)+" genes\n")
        
        gene_SFS = np.random.poisson((gene_length_temp/genome_length)*sample_SFS_temp)
#        foutput.write("shape of gene SFS = "+str(np.shape(gene_SFS)))
        # ignore fixed sites/monomorphic sites for now
        gene_counter = (np.arange(sample_size_temp-1)+1)
        gene_freq = (np.arange(sample_size_temp-1)+1)/sample_size_temp
#        foutput.write("shape of frequency array = "+str(np.shape(gene_freq)))
        # compute xbar
        total_counts_gene[gene] = np.sum(gene_counter*gene_SFS)
        xbar_gene[gene] = np.sum(gene_freq*gene_SFS)
        # compute x2bar
        x2bar_gene[gene] = np.sum(gene_freq*gene_freq*gene_SFS)
        #compute pi
        pi_gene[gene] = 2*(xbar_gene[gene]-x2bar_gene[gene])
        # create Log(1+SFS)
        log1plusSFS = np.log1p(gene_SFS)
        # sum to make logstat
        logstat_gene[gene] = np.sum(log1plusSFS)
        # create Log(1+SFS/U)
        log1plusSFSUscale = np.log1p(gene_SFS/Ugene)
        # sum to make logstatUscale
        logstatUscale_gene[gene] = np.sum(log1plusSFSUscale)
        # create Log(1+SFS/U)
        log1plusSFSMUscale = np.log1p(gene_SFS/MUgene)
        # sum to make logstatMUscale
        logstatMUscale_gene[gene] = np.sum(log1plusSFSMUscale)
        # create Log(1+SFS/U)
        log1plusSFSLscale = np.log1p(gene_SFS/Lgene)
        # sum to make logstatLscale
        logstatLscale_gene[gene] = np.sum(log1plusSFSLscale)
        
        # create frequency
        x_SFS=(1+np.arange(gene_SFS.size))/sample_size_temp
        # sum to make logload = x*log(1+phi)
        logload_gene[gene] = np.sum(x_SFS*log1plusSFS)
        # sum to make logload2 = x^2*log(1+phi)
        logload2_gene[gene] = np.sum(x_SFS*x_SFS*log1plusSFS)
        
        # compute segregating sites
        segregating_sites_gene[gene] = np.sum(gene_SFS)
        # compute singletons
        singletons_gene[gene] = gene_SFS[0]
        # compute fraction of rare
        frac_rare_gene[gene] = np.sum(gene_SFS[:rare_number_temp])/segregating_sites_gene[gene]
    
    
    
        if print_each_gene_SFS_flag_temp==1:
            assure_path_exists("each_gene_SFS")
            os.chdir("each_gene_SFS")
            fixed_mutations_temp=0
            file_prefix="singleGeneSFS"
            gene_seed=gene+seedstart
            print_raw_SFS_flag_temp=1
            hist_instead_of_counts_flag_temp=1
            irrelevant_hist, irrelevant_singleton = print_raw_SFS(initpopsize_temp, sample_size_temp, gene_SFS, fixed_mutations_temp, demography_type,ancestry_type, s_temp, h_temp, mutation_rate_temp,gene_length_temp, growth_rate_temp, sparse_temp, linearS,linearL,linearU, file_prefix, sample_size_temp, gene_seed, print_raw_SFS_flag_temp, hist_instead_of_counts_flag_temp)
            os.chdir("..")
    
    
    return total_counts_gene, xbar_gene, x2bar_gene, pi_gene, logstat_gene,logstatUscale_gene,logstatMUscale_gene,logstatLscale_gene, logload_gene, logload2_gene, segregating_sites_gene, singletons_gene, frac_rare_gene




#_______________________________________________________________
#
#                  IMPORT GENE LENGTH LIST
#_______________________________________________________________

def import_gene_length_list(length_list_name, damaging_or_synon, foutput):
    
    path="/Users/dbalick/Documents/_MY_SOFTWARE/simDoSe/_simDoSe_v"+simdose_version+"/_gene_length_lists"
    os.chdir(path)


    filename_gene_length_TSV = "gene_lengths_"+length_list_name+".tsv"
    
    length_list_DF = pd.read_csv(filename_gene_length_TSV, index_col=None, na_values = {"#N/A","-999"}, delimiter="\t")
    
    if damaging_or_synon=="damaging":
        length_list_temp = np.array(length_list_DF["LOF_probably"].dropna())
#        length_list_temp = length_list_DF["LOF_probably"].values
#        length_list_temp = np.array(length_list_temp.dropna)
    elif damaging_or_synon=="synon":
        length_list_temp = np.array(length_list_DF["synon"].dropna())
#        length_list_temp = length_list_DF["synon"].values
#        length_list_temp = np.array(length_list_temp.dropna)
    else:
        length_list_temp = np.array(length_list_DF["length"].dropna())


#    length_list_int_temp=np.int(length_list_temp)
    length_list_int_temp=length_list_temp.astype(int)


    foutput.write("\n\nImporting gene list "+length_list_name+"\n")
#    foutput.write("\n Llist_import= "+str(length_list_int_temp[1:10])+"\n")


    return length_list_int_temp











#_______________________________________________________________
#
#                  PRINTING RAW/SAMPLE SFS
#_______________________________________________________________


def print_raw_SFS(initpopsize_temp, popsize_temp, counts_temp, fixed_mutations_temp, demography_type,ancestry_type, s_temp, h_temp, mu_temp,L_temp, growth_rate_temp, sparse_temp, linearS,linearL,linearU, filename_prefix, sample_size_temp, seed_temp, printrawSFS_flag_temp, hist_instead_of_counts_flag):
    
    
    if sample_size_temp!=1 and sample_size_temp!=popsize_temp:
        foutput.write("\n\n ERROR:  SAMPLE SIZE AND POP SIZE NOT LINING UP FOR SAMPLE SFS. \n\n")
    
    # Log transform simulation numbers for file title
    if linearL==1:
        logL=L_temp
    else:
        logL = round(np.log10(L_temp),1)
    if linearU==1:
        logmu = float(mu_temp)
    else:
        logmu = round(np.log10(mu_temp),5)
    if initpopsize_temp%10==0:
        logN = round(np.log10(initpopsize_temp),1)
    else:
        logN=initpopsize_temp
    s_is_beneficial=0
    if s_temp < 0:
        logS = round(np.log10(np.abs(s_temp)),1)
    elif s_temp > 0:
        logS = round(np.log10(s_temp),1)
        s_is_beneficial = 1
    elif s_temp==0:
        logS="NEUTRAL"


    if hist_instead_of_counts_flag==1:
        hist=counts_temp

    else:

        if bincount_from_numpy==0:
            bin_range = np.arange(popsize_temp+1)+0.5
            hist, bin_edges = np.histogram(counts_temp, bin_range)
        
        elif bincount_from_numpy==1:
            ###   THIS IS ANOTHER WAY TO MAKE THE HISTOGRAM (FROM DMJ)
            counts_int = counts_temp.astype(int)
            hist = np.bincount(counts_int, minlength=(popsize_temp+1))
            hist = hist[1:popsize_temp]

    if s_is_beneficial==1:
        pos_selection_label="positive"
    else:
        pos_selection_label=""
    
    N_label="_2N_"+str(logN)
    S_label="_S"+pos_selection_label+"_"+str(logS)
    H_label="_h_"+str(h_temp)
    Mu_label="_mu_"+str(logmu)
    L_label="_L_"+str(logL)

    if hist_instead_of_counts_flag==1:
        L_label="_L_"+str(L_temp)

    
    growth_label=""
    ancestry_label=""
    sample_size_label=""
    seed_label = ""
    if seed_temp!="XX":
        seed_label="_run_"+str(seed_temp)

    if demography_type=="linear" or demography_type=="exponential" or demography_type=="supertennessen" or demography_type=="gaussian":
        growth_label="_growth_"+str(growth_rate_temp)
    if demography_type=="browning":
        if growth_rate_temp==0.0195:
            IBD_growth_label="0.0833"
        else:
            IBD_growth_label=str(growth_rate_temp)
        growth_label="_growth_"+IBD_growth_label


    if demography_type=="tennessen" or demography_type=="supertennessen" or demography_type=="gaussian" or demography_type=="browning":
        ancestry_label="_"+ancestry_type
        if int(initpopsize_temp)==28948:
            N_label=""
    if linearS:
        S_label= "_Slinear_"+pos_selection_label+"_"+str(s_temp)

    if sample_size_temp!=1:
        sample_size_label="_samplesize_"+str(sample_size_temp)


    if hist_instead_of_counts_flag==1:
        monomorphic_counts=fixed_mutations_temp
    else:
        monomorphic_counts = L_temp-np.sum(hist[:(popsize_temp-1)])
    singleton_count_temp=hist[0]


    if printrawSFS_flag_temp==1:

        sfsfile = open(filename_prefix+"_"+demography_type + ancestry_label + sample_size_label+ N_label+S_label+H_label+Mu_label+L_label+growth_label+seed_label+".tsv", "w")

        sfsfile.write("Counts\tNumber_of_alleles\n")
    #    ###  DISPLAY HISTOGRAM IF DESIRED
    #    if show_plot==1:
    #        plt.bar(hist, bin_edges[:-1])
    #        plt.show()
        sfsfile.write( "0"+"\t"+str(monomorphic_counts)+"\n")
        for k in range(len(hist)-1):
            if sparse_temp and hist[k]>0.5:
                sfsfile.write( str(k+1)+"\t"+str(hist[k])+"\n")
            elif not sparse_temp:
                sfsfile.write( str(k+1)+"\t"+str(hist[k])+"\n")
    #    sfsfile.write("Fixations"+"\t"+str(fixed_mutations_temp)+"\n")
        sfsfile.close()


    return hist, singleton_count_temp



#_______________________________________________________________
#
#              PRINTING SUMMARY STATS FOR RAW/SAMPLE SFS
#_______________________________________________________________



def print_summary_stats(initpopsize_temp, popsize_temp, counts_temp, fixed_mutations_temp, demography_type, ancestry_type,s_temp, h_temp, complete_underdom_temp, mu_temp,L_temp, singleton_count_temp, growth_rate_temp, linearS,linearL,linearU, filename_prefix,sample_size_temp, final_popsize_temp, seed_temp):
    
#    care_about_growth_rate = 0 # add growth rate to filename?

    counts_temp=counts_temp[counts_temp>0.5]
    
    if sample_size_temp==1:
        counts_temp=counts_temp[counts_temp<popsize_temp]
        frequency_temp = counts_temp/popsize_temp # frequency is defined by true population size
        xbar_from_counts = np.sum(counts_temp)/popsize_temp
        x2bar_from_counts = np.sum(counts_temp*counts_temp)/(popsize_temp*popsize_temp)
    elif sample_size_temp==popsize_temp:
        counts_temp=counts_temp[counts_temp<sample_size_temp]
        frequency_temp = counts_temp/sample_size_temp # frequency is defined by sample size
        xbar_from_counts = np.sum(counts_temp)/(sample_size_temp)
        x2bar_from_counts = np.sum(counts_temp*counts_temp)/(sample_size_temp*sample_size_temp)

    else:
        foutput.write("\n\n ERROR:  SAMPLE SIZE AND POP SIZE NOT LINING UP FOR SAMPLE SFS. \n\n")
    xbar = np.sum(frequency_temp)
    x2bar = np.sum(frequency_temp*frequency_temp)
    pi = 2*(xbar-x2bar)
    segregating_sites = counts_temp.size
    #    theta = 2*popsize_temp*mu_temp*L_temp  ##  theta for initial or final popsize for non-equilib?
    theta = 2*initpopsize_temp*mu_temp*L_temp
    ####  NOW COMPUTE STATS IN TERMS OF COUNTS, RATHER THAN MOMENTS
    # first add monomorphic
    if counts_temp.size>L_temp:  #THIS SHOULD BE UNNECCESARY
        counts_temp=counts_temp[counts_temp>0.5]
        if sample_size_temp==popsize_temp:
            counts_temp=counts_temp[counts_temp<sample_size_temp]
            monomorphic = np.zeros((L_temp-segregating_sites))
        elif sample_size_temp==1:
            counts_temp=counts_temp[counts_temp<popsize_temp]
            monomorphic = np.zeros((L_temp-segregating_sites))
        else:
            foutput.write("ERROR: NUMBER OF SEG SITES > LGENOME")
            monomorphic = np.array([])
    else:
        monomorphic = np.zeros((L_temp-segregating_sites))

    counts_temp_with_monomorphic = np.append(counts_temp, monomorphic)
    mean_of_counts = np.mean(counts_temp_with_monomorphic)
    var_of_counts = np.var(counts_temp_with_monomorphic)

    
    
    
    # Log transform simulation numbers for file title
    if linearL ==1:
        logL=L_temp
    else:
        logL = round(np.log10(L_temp),1)
    if linearU==1:
        logmu = float(mu_temp)
    else:
        logmu = round(np.log10(mu_temp),5)
    if initpopsize_temp%10==0:
        logN = round(np.log10(initpopsize_temp),1)
    else:
        logN=initpopsize_temp
    s_is_beneficial=0
    if s_temp < 0:
        logS = round(np.log10(np.abs(s_temp)),1)
    elif s_temp > 0:
        logS = round(np.log10(s_temp),1)
        s_is_beneficial = 1
    elif s_temp==0:
        logS="NEUTRAL"


    if s_is_beneficial==1:
        pos_selection_label="positive"
    else:
        pos_selection_label=""
    
    N_label="_2N_"+str(logN)
    S_label="_S"+pos_selection_label+"_"+str(logS)
    H_label="_h_"+str(h_temp)
    Mu_label="_mu_"+str(logmu)
    L_label="_L_"+str(logL)
    
    growth_label=""
    ancestry_label=""
    sample_size_label=""
    seed_label = ""
    if seed_temp!="XX":
        seed_label="_run_"+str(seed_temp)


    if demography_type=="linear" or demography_type=="exponential" or demography_type=="supertennessen" or demography_type=="gaussian":
#        care_about_growth_rate=1
        growth_label="_growth_"+str(growth_rate_temp)

    if demography_type=="browning":
        if growth_rate_temp==0.0195:
            IBD_growth_label="0.0833"
        else:
            IBD_growth_label=str(growth_rate_temp)
        growth_label="_growth_"+IBD_growth_label


    if demography_type=="tennessen" or demography_type=="supertennessen" or demography_type=="gaussian" or demography_type=="browning" :
        ancestry_label="_"+ancestry_type
        if int(initpopsize_temp)==28948:
            N_label=""

    if linearS:
        S_label= "_Slinear_"+pos_selection_label+"_"+str(s_temp)

    if sample_size_temp!=1:
        sample_size_label="_samplesize_"+str(sample_size_temp)

    summary_stats_file = open(filename_prefix+"_"+demography_type + ancestry_label+sample_size_label + N_label+S_label+H_label+Mu_label+L_label+growth_label+seed_label+".txt", "w")




    if sample_size_temp==1:

        summary_stats_file.write("##########################\n")
        summary_stats_file.write("#   SUMMARY STATISTICS   #\n")
        summary_stats_file.write("##########################\n\n")

    else:

        summary_stats_file.write("#################################\n")
        summary_stats_file.write("#   SAMPLE SUMMARY STATISTICS   #\n")
        summary_stats_file.write("#################################\n\n")
    



    summary_stats_file.write("Demography = "+"\t"+str(demography_type)+"\n")
    if demography_type=="tennessen" or demography_type=="supertennessen" or demography_type=="gaussian" or demography_type=="browning":
        summary_stats_file.write("Ancestry = "+"\t"+str(ancestry_type)+"\n")

    summary_stats_file.write("initial size 2N (haploid) = "+"\t"+str(initpopsize_temp)+"\n")
    summary_stats_file.write("Final size 2N (haploid) = "+"\t"+str(final_popsize_temp)+"\n")
    if sample_size_temp!=1:
        summary_stats_file.write("Sample size 2M (haploid) = "+"\t"+str(sample_size_temp)+"\n")
    summary_stats_file.write("mu (per base) = "+"\t"+str(mu_temp)+"\n")
    summary_stats_file.write("L (number of bases) = "+"\t"+str(L_temp)+"\n")
    summary_stats_file.write("U (total mut. rate per indiv.) = "+ str(mu_temp*L_temp)+"\n")
    if s_is_beneficial:
        summary_stats_file.write("s (selection is beneficial) = "+"\t"+str(s_temp)+"\n")
    elif not s_is_beneficial:
        summary_stats_file.write("s (selection is deleterious) = "+"\t"+str(s_temp)+"\n")

#    if care_about_growth_rate:
    summary_stats_file.write("growth rate per generation = "+"\t"+str(growth_rate_temp)+"\n")



    summary_stats_file.write("h = "+"\t"+str(h_temp)+"\n")
    if complete_underdom_temp:
        summary_stats_file.write("RUNNING COMPLETE UNDERDOMINANCE: s_het=hs, s_hom=0"+"\n")
    summary_stats_file.write("theta (2(2N_initial) mu L) = "+"\t"+str(theta)+"\n\n")

    summary_stats_file.write("\n======================================================\n")
    summary_stats_file.write("\nMOMENTS (Computed without monomorphic/fixed alleles)\n")

    summary_stats_file.write("mutation burden xbar = "+"\t"+str(xbar)+"\n")
    summary_stats_file.write("homozygosity x2bar = "+"\t"+str(x2bar)+"\n")
    summary_stats_file.write("heterozygosity pi = "+"\t"+str(pi)+"\n")
    summary_stats_file.write("segregating sites = "+"\t"+str(segregating_sites)+"\n")
    summary_stats_file.write("num singletons = "+"\t"+str(singleton_count_temp)+"\n")
    summary_stats_file.write("singletons/seg_sites = "+"\t"+str(singleton_count_temp/segregating_sites)+"\n")
    summary_stats_file.write("fixations = "+"\t"+str(fixed_mutations_temp)+"\n\n")

#
#    # Double check to make sure xbar and x2bar are ok...
#    summary_stats_file.write("_________Double check moments from counts ______________\n")
#    summary_stats_file.write("xbar from counts = "+"\t"+str(xbar_from_counts)+"\n")
#    summary_stats_file.write("x2bar from counts = "+"\t"+str(x2bar_from_counts)+"\n\n")
#
#



#    ########################################################################
#    ###############  POISSON STATS FOR PTV PAPER   #########################
#    ########################################################################
#    summary_stats_file.write("_________Count statistics over variants______________\n")
#    summary_stats_file.write("NOTE: this includes monomorphic sites in average and var \n")
#    summary_stats_file.write("mean of counts = "+"\t"+str(mean_of_counts)+"\n")
#    summary_stats_file.write("var of counts = "+"\t"+str(var_of_counts)+"\n")
#    summary_stats_file.write("_____Poisson assumption test (=1 for Poisson)_____\n")
#    summary_stats_file.write("var/mean of counts = "+"\t"+str(var_of_counts/mean_of_counts)+"\n")


    summary_stats_file.write("\n\n\n--- Total runtime: %s minutes ---\n" % ((time.time() - start_time)/60))


    summary_stats_file.close()

    return


#_______________________________________________________________
#
#              PRINTING SIMULATED GENES
#_______________________________________________________________

def print_simulated_genes(number_genes_temp, gene_length_temp, gene_total_counts_temp, gene_xbar_temp, gene_x2bar_temp, gene_pi_temp, gene_logstat_temp,gene_logstatUscale_temp,gene_logstatMUscale_temp,gene_logstatLscale_temp,gene_logload_temp, gene_logload2_temp,gene_segregating_sites_temp, gene_singletons_temp,gene_frac_rare_temp,rare_number_temp, initpopsize_temp, demography_type, ancestry_type, s_temp, h_temp, mu_temp, L_temp, growth_rate_temp, sample_size_temp, data_source, linearS,linearL,linearU,filename_prefix, seed_temp, final_popsize_temp, print_genes_flag, print_gene_SFS_flag, sparse_temp, length_list_temp, length_list_name_temp, print_each_gene_SFS_flag, geneSFS):

#    care_about_growth_rate = 0 # add growth rate to filename?
    print_individual_genes = print_genes_flag # print summary stats for individual genes
    print_SFS_over_sim_genes = print_gene_SFS_flag # print SFS of xbar only over individual genes
#
#    theta = 2*initpopsize_temp*mu_temp*L_temp
#    theta_final = 2*final_popsize_temp*mu_temp*L_temp
#    theta_biallelic = 2*initpopsize_temp*mu_temp*L_gene
#    theta_final_biallelic = 2*final_popsize_temp*mu_temp*L_gene

    # Log transform simulation numbers for file title
    logNgenes=round(np.log10(number_genes_temp),1)
    logLgene = round(np.log10(gene_length_temp),1)
    if linearL==1:
        logL=L_temp
    else:
        logL = round(np.log10(L_temp),1)
    if linearU==1:
        logmu=float(mu_temp)
    else:
        logmu = round(np.log10(mu_temp),5)
    if initpopsize_temp%10==0:
        logN = round(np.log10(initpopsize_temp),1)
    else:
        logN=initpopsize_temp
    s_is_beneficial=0
    if s_temp < 0:
        logS = round(np.log10(np.abs(s_temp)),1)
    if s_temp > 0:
        logS = round(np.log10(s_temp),1)
        s_is_beneficial = 1
    elif s_temp==0:
        logS="NEUTRAL"

    if s_is_beneficial==1:
        pos_selection_label="positive"
    else:
        pos_selection_label=""
    
    N_label="_2N_"+str(logN)
    S_label="_S"+pos_selection_label+"_"+str(logS)
    H_label="_h_"+str(h_temp)
    Mu_label="_mu_"+str(logmu)
    L_label="_L_"+str(logL)
    if length_list_name_temp=="XX":
        Lgene_label = "_Lgene_"+str(logLgene)
    else:
        Lgene_label = "_Lgene_"+length_list_name_temp
    Ngenes_label = "_Ngenes_"+str(logNgenes)
    
    growth_label=""
    ancestry_label=""
    sample_size_label=""

    if sample_size_temp!=1:
        sample_size_label="_samplesize_"+str(sample_size_temp)

    if demography_type=="linear" or demography_type=="exponential" or demography_type=="supertennessen" or demography_type=="gaussian":
        growth_label="_growth_"+str(growth_rate_temp)

    if demography_type=="browning":
        if growth_rate_temp==0.0195:
            IBD_growth_label="0.0833"
        else:
            IBD_growth_label=str(growth_rate_temp)
        growth_label="_growth_"+IBD_growth_label


    if demography_type=="tennessen" or demography_type=="supertennessen" or demography_type=="gaussian" or demography_type=="browning":
        ancestry_label="_"+ancestry_type
        if int(initpopsize_temp)==28948:
            N_label=""
    if linearS==1:
        S_label= "_Slinear_"+pos_selection_label+"_"+str(s_temp)
    if data_source=="exac" and ancestry_type=="european" and sample_size_temp==68858:
        sample_size_label="_exac"

    seed_label=""
    if seed_temp!="XX":
        seed_label="_run_"+seed_temp


    #compute average xbar var xbar over Ngenes
    avg_counts_simgenes = np.mean(gene_total_counts_temp)
    avg_xbar_simgenes = np.mean(gene_xbar_temp)
    var_counts_simgenes = np.var(gene_total_counts_temp)
    var_xbar_simgenes = np.var(gene_xbar_temp)



    if print_individual_genes==1:

        if print_SFS_over_sim_genes==1:
            assure_path_exists("SFS_over_biallelic_genes")
            os.chdir("SFS_over_biallelic_genes")
            simgenes_file = open(filename_prefix+"_"+demography_type + ancestry_label + sample_size_label+Ngenes_label+Lgene_label+ N_label+S_label+H_label+Mu_label+L_label+growth_label+seed_label+".tsv", "w")


            #######  This version does not have logstat stuff... swap with commented below if you want logstat
            simgenes_file.write("length\ttotal_counts\txbar\tx2bar\tpi\tseg_sites\tsingletons\tfrac_rare\n")
            if length_list_name_temp=="XX":
                for g in range(gene_xbar_temp.size):
                    simgenes_file.write( str(gene_length_temp)+"\t"+str(gene_total_counts_temp[g])+"\t"+str(gene_xbar_temp[g])+"\t"+str(gene_x2bar_temp[g])+"\t"+str(gene_pi_temp[g])+"\t"+str(gene_segregating_sites_temp[g])+"\t"+str(gene_singletons_temp[g])+"\t"+str(gene_frac_rare_temp[g])+"\n")
            else:
                for g in range(length_list_temp.size):
                    simgenes_file.write( str(length_list_temp[g])+"\t"+str(gene_total_counts_temp[g])+"\t"+str(gene_xbar_temp[g])+"\t"+str(gene_x2bar_temp[g])+"\t"+str(gene_pi_temp[g])+"\t"+str(gene_segregating_sites_temp[g])+"\t"+str(gene_singletons_temp[g])+"\t"+str(gene_frac_rare_temp[g])+"\n")
            simgenes_file.close()
            os.chdir("..")

        else:
        
            simgenes_file = open(filename_prefix+"_"+demography_type + ancestry_label + sample_size_label+Ngenes_label+Lgene_label+ N_label+S_label+H_label+Mu_label+L_label+growth_label+seed_label+".tsv", "w")


            #######  This version does not have logstat stuff... swap with commented below if you want logstat
            simgenes_file.write("length\txbar\tx2bar\tpi\tseg_sites\tsingletons\tfrac_rare\n")
            if length_list_name_temp=="XX":
                for g in range(gene_xbar_temp.size):
                    simgenes_file.write( str(gene_length_temp)+"\t"+str(gene_xbar_temp[g])+"\t"+str(gene_x2bar_temp[g])+"\t"+str(gene_pi_temp[g])+"\t"+str(gene_segregating_sites_temp[g])+"\t"+str(gene_singletons_temp[g])+"\t"+str(gene_frac_rare_temp[g])+"\n")
            else:
                for g in range(length_list_temp.size):
                    simgenes_file.write( str(length_list_temp[g])+"\t"+str(gene_xbar_temp[g])+"\t"+str(gene_x2bar_temp[g])+"\t"+str(gene_pi_temp[g])+"\t"+str(gene_segregating_sites_temp[g])+"\t"+str(gene_singletons_temp[g])+"\t"+str(gene_frac_rare_temp[g])+"\n")
            simgenes_file.close()

        


#######  Version below includes logstat stuff turn back on if you want logstat #####
        
#        simgenes_file.write("length\ttotal_counts\txbar\tx2bar\tpi\tlogstat\tlogstatUscale\tlogstatMUscale\tlogstatLscale\tlogload\tlogload2\tseg_sites\tsingletons\tfrac_rare\n")
#        if length_list_name_temp=="XX":
#            for g in range(gene_xbar_temp.size):
#                simgenes_file.write( str(gene_length_temp)+"\t"+str(gene_total_counts_temp[g])+"\t"+str(gene_xbar_temp[g])+"\t"+str(gene_x2bar_temp[g])+"\t"+str(gene_pi_temp[g])+"\t"+str(gene_logstat_temp[g])+"\t"+str(gene_logstatUscale_temp[g])+"\t"+str(gene_logstatMUscale_temp[g])+"\t"+str(gene_logstatLscale_temp[g])+"\t"+str(gene_logload_temp[g])+"\t"+str(gene_logload2_temp[g])+"\t"+str(gene_segregating_sites_temp[g])+"\t"+str(gene_singletons_temp[g])+"\t"+str(gene_frac_rare_temp[g])+"\n")
#        else:
#            for g in range(length_list_temp.size):
#                simgenes_file.write( str(length_list_temp[g])+"\t"+str(gene_total_counts_temp[g])+"\t"+str(gene_xbar_temp[g])+"\t"+str(gene_x2bar_temp[g])+"\t"+str(gene_pi_temp[g])+"\t"+str(gene_logstat_temp[g])+"\t"+str(gene_logstatUscale_temp[g])+"\t"+str(gene_logstatMUscale_temp[g])+"\t"+str(gene_logstatLscale_temp[g])+"\t"+str(gene_logload_temp[g])+"\t"+str(gene_logload2_temp[g])+"\t"+str(gene_segregating_sites_temp[g])+"\t"+str(gene_singletons_temp[g])+"\t"+str(gene_frac_rare_temp[g])+"\n")




    if print_SFS_over_sim_genes==1:
        
        assure_path_exists("SFS_over_biallelic_genes")
        os.chdir("SFS_over_biallelic_genes")

        simgenes_SFS_file = open("pooledSFS_"+filename_prefix+"_"+demography_type + ancestry_label + sample_size_label+Ngenes_label+Lgene_label+ N_label+S_label+H_label+Mu_label+L_label+growth_label+seed_label+".tsv", "w")
            
        
        # Make a histogram of pooled counts per gene (genic SFS)
        gene_total_counts_int = gene_total_counts_temp.astype(int)
        pooled_gene_hist = np.bincount(gene_total_counts_int, minlength=(sample_size_temp+1))
        # this should remove monomorphic sites on both ends.
        pooled_gene_hist = pooled_gene_hist[1:sample_size_temp]
#        
#        #compute average xbar var xbar over Ngenes
#        avg_counts_simgenes = np.mean(gene_total_counts_int)
#        avg_xbar_simgenes = np.mean(gene_xbar_temp)
#        var_counts_simgenes = np.var(gene_total_counts_int)
#        var_xbar_simgenes = np.var(gene_xbar_temp)
#        

        monomorphic_genes = number_genes_temp-np.sum(pooled_gene_hist[:(sample_size_temp-1)])

        # Print genic SFS
        simgenes_SFS_file.write("POOLED_gene_Counts\tNumber_of_genes\n")
        simgenes_SFS_file.write( "0"+"\t"+str(monomorphic_genes)+"\n")
        for k in range(len(pooled_gene_hist)-1):  # vector is length popsize long, but we only want seg_sites
            if sparse_temp and pooled_gene_hist[k]>0.5:
                simgenes_SFS_file.write( str(k+1)+"\t"+str(pooled_gene_hist[k])+"\n")
            elif not sparse_temp:
                simgenes_SFS_file.write( str(k+1)+"\t"+str(pooled_gene_hist[k])+"\n")
#                # Ignore fixed mutations for now
#                simgenes_SFS_file.write("Fixations"+"\t"+str(fixed_mutations_temp)+"\n")

        simgenes_SFS_file.close()
        os.chdir("..")
            
    

    ################################################################################
    #
    #   NOW MAKE SUMMARY STATS FOR POOLED GENES (BIALLELIC ASSUMPTION)
    #
    ################################################################################

    # Compute the number of genes with segregating total counts (biallelic seg_sites)
    pooled_segregating_list = gene_total_counts_temp[gene_total_counts_temp > 0.5]
    pooled_segregating_list = pooled_segregating_list[pooled_segregating_list<sample_size_temp]
    pooled_segregating = pooled_segregating_list.size
    # Compute the number of genes with a singleton total count
    pooled_singleton_list = pooled_segregating_list[pooled_segregating_list==1]
    pooled_doubleton_list = pooled_segregating_list[pooled_segregating_list==2]
    pooled_tripleton_list = pooled_segregating_list[pooled_segregating_list==3]
    pooled_quadrupleton_list = pooled_segregating_list[pooled_segregating_list==4]
    pooled_singletons = pooled_singleton_list.size
    pooled_doubletons = pooled_doubleton_list.size
    pooled_tripletons = pooled_tripleton_list.size
    pooled_quadrupletons = pooled_quadrupleton_list.size



    ##### First print genes to file #####
    
    if print_each_gene_SFS_flag==1:
        if print_SFS_over_sim_genes==1:
            assure_path_exists("SFS_over_biallelic_genes/per_gene_SFS")
            os.chdir("SFS_over_biallelic_genes/per_gene_SFS")
        else:
            assure_path_exists("per_gene_SFS")
            os.chdir("per_gene_SFS")
        for gene in range(number_genes_temp):

            fixed_mutations_temp=0
            file_prefix="singleGeneSFS_gene"+str(gene)
            print_raw_SFS_flag_temp=1
            hist_instead_of_counts_flag_temp=1
            irrelevant_hist, irrelevant_singleton = print_raw_SFS(initpopsize_temp, sample_size_temp, geneSFS[gene], fixed_mutations_temp, demography_type,ancestry_type, s_temp, h_temp, mu_temp, gene_length_temp, growth_rate_temp, sparse_temp, linearS,linearL,linearU, file_prefix, sample_size_temp, seed_temp, print_raw_SFS_flag_temp, hist_instead_of_counts_flag_temp)
        if print_SFS_over_sim_genes==1:
            os.chdir("../..")
        else:
            os.chdir("..")
    
    

#    if print_SFS_over_sim_genes==1:
#
#        assure_path_exists("simulated_genes_summary_stats")
#        os.chdir("simulated_genes_summary_stats")
#
#    else:
#
#        assure_path_exists("simulated_genes_summary_stats")
#        os.chdir("simulated_genes_summary_stats")


    if print_SFS_over_sim_genes==1:
        assure_path_exists("SFS_over_biallelic_genes/simulated_genes_summary_stats")
        os.chdir("SFS_over_biallelic_genes/simulated_genes_summary_stats")
    else:
        assure_path_exists("simulated_genes_summary_stats")
        os.chdir("simulated_genes_summary_stats")

#    assure_path_exists("simulated_genes_summary_stats")
#    os.chdir("simulated_genes_summary_stats")


    simgenes_summary_stats_file = open(filename_prefix+"_stats_"+demography_type + ancestry_label + sample_size_label+Ngenes_label+Lgene_label+ N_label+S_label+H_label+Mu_label+L_label+growth_label+seed_label+".tsv", "w")

    simgenes_summary_stats_file.write("######################################################\n")
    simgenes_summary_stats_file.write("#   SUMMARY STATISTICS FOR GROUP OF SIMULATED GENES  #\n")
    simgenes_summary_stats_file.write("######################################################\n\n")
    if print_SFS_over_sim_genes==1:
        simgenes_summary_stats_file.write("BIALLELIC SUMMARY STATS (POOL SITES)\n\n")
    else:
        simgenes_summary_stats_file.write("SUMMARY STATS FOR POOLED SITES PER GENE\n\n")
    simgenes_summary_stats_file.write("Demography = "+"\t"+str(demography_type)+"\n")
    if demography_type=="tennessen" or demography_type=="supertennessen" or demography_type=="gaussian" or demography_type=="browning":
        simgenes_summary_stats_file.write("Ancestry = "+"\t"+str(ancestry_type)+"\n")

    simgenes_summary_stats_file.write("initial size 2N (haploid) = "+"\t"+str(initpopsize_temp)+"\n")
    simgenes_summary_stats_file.write("Final size 2N (haploid) = "+"\t"+str(final_popsize_temp)+"\n")
    simgenes_summary_stats_file.write("Growth rate (exponent) = "+"\t"+str(growth_rate_temp)+"\n")
    if sample_size_temp!=1:
        simgenes_summary_stats_file.write("Sample size 2M (haploid) = "+"\t"+str(sample_size_temp)+"\n")
    simgenes_summary_stats_file.write("mu (per base) = "+"\t"+str(mu_temp)+"\n")
    simgenes_summary_stats_file.write("L (number of bases) = "+"\t"+str(L_temp)+"\n")

    simgenes_summary_stats_file.write("U (total mut. rate per indiv.) = "+ str(mu_temp*L_temp)+"\n")
    simgenes_summary_stats_file.write("L_gene (number of pooled bases per gene) = "+"\t"+str(gene_length_temp)+"\n")
    
    simgenes_summary_stats_file.write("U_gene (total genic mut. rate per indiv.--biallelic mu rate) = "+ str(mu_temp*gene_length_temp)+"\n")

    simgenes_summary_stats_file.write("Number of genes simulated = "+ str(number_genes_temp)+"\n")


    if s_is_beneficial:
        simgenes_summary_stats_file.write("s (selection is beneficial) = "+"\t"+str(s_temp)+"\n")
    elif not s_is_beneficial:
        simgenes_summary_stats_file.write("s (selection is deleterious) = "+"\t"+str(s_temp)+"\n")



    simgenes_summary_stats_file.write("h = "+"\t"+str(h_temp)+"\n")
#    simgenes_summary_stats_file.write("theta (2(2N_initial) mu L) = "+"\t"+str((final_popsize_temp*))+"\n\n")
#    simgenes_summary_stats_file.write("biallelic theta (2(2N_initial) mu L_gene) = "+"\t"+str(theta)+"\n\n")


    if print_SFS_over_sim_genes:
        simgenes_summary_stats_file.write("\n=======================================================\n")
        simgenes_summary_stats_file.write("\nBIALLELIC MOMENTS (Computed without monomorphic/fixed alleles)\n")


    #    mean_frequency_biallelic = np.mean(gene_xbar_temp)
    #
    #    simgenes_summary_stats_file.write("mean biallelic frequency = "+"\t"+str(mean_frequency_biallelic)+"\n")


        # Print mean and var of counts including monomorphic
        simgenes_summary_stats_file.write("\n\nMean of biallelic counts (number of individuals with mutations per gene) = "+"\t"+str(avg_counts_simgenes)+"\n")
        simgenes_summary_stats_file.write("Var of biallelic counts (var individuals with mutations per gene) = "+"\t"+str(var_counts_simgenes)+"\n")
        simgenes_summary_stats_file.write("Var/Mean of biallelic counts per gene = "+"\t"+str(var_counts_simgenes/avg_counts_simgenes)+"\n\n")


        # Print mean and var of xbar including monomorphic instead of mean and var of counts
        simgenes_summary_stats_file.write("\n\nMean of biallelic frequency (frac of individuals with mutations per gene) = "+"\t"+str(avg_xbar_simgenes)+"\n")
        simgenes_summary_stats_file.write("Var of biallelic frequency = "+"\t"+str(var_xbar_simgenes)+"\n")
        simgenes_summary_stats_file.write("(Var*sample_size^2)/(Mean*sample_size) of biallelic frequency per gene = "+"\t"+str(var_xbar_simgenes*sample_size_temp/avg_xbar_simgenes)+"\n\n\n")



        # Print stats related to fraction of singletons
        simgenes_summary_stats_file.write("number of segregating biallelic sites (genes with >0 seg_sites) = "+"\t"+str(pooled_segregating)+"\n")

        simgenes_summary_stats_file.write("number of singleton biallelic sites (genes with xbar=1/sample_size) = "+"\t"+str(pooled_singletons)+"\n")

        simgenes_summary_stats_file.write("number of doubleton biallelic sites (genes with xbar=2/sample_size) = "+"\t"+str(pooled_doubletons)+"\n")

        simgenes_summary_stats_file.write("number of tripleton biallelic sites (genes with xbar=3/sample_size) = "+"\t"+str(pooled_tripletons)+"\n")

        simgenes_summary_stats_file.write("number of quadrupleton biallelic sites (genes with xbar=4/sample_size) = "+"\t"+str(pooled_quadrupletons)+"\n")

        if pooled_segregating >0:
            simgenes_summary_stats_file.write("fraction of singleton biallelic sites (genes with xbar=1/sample_size) = "+"\t"+str(pooled_singletons/pooled_segregating)+"\n")

            simgenes_summary_stats_file.write("fraction of doubleton biallelic sites (genes with xbar=2/sample_size) = "+"\t"+str(pooled_doubletons/pooled_segregating)+"\n")

            simgenes_summary_stats_file.write("combined fraction of singleton and doubleton biallelic sites = "+"\t"+str((pooled_singletons+ pooled_doubletons)/pooled_segregating)+"\n")

            simgenes_summary_stats_file.write("combined fraction of singleton, doubleton, and tripleton biallelic sites = "+"\t"+str((pooled_singletons+ pooled_doubletons + pooled_tripletons)/pooled_segregating)+"\n")
            simgenes_summary_stats_file.write("combined fraction of singleton-to-quadrupleton biallelic sites = "+"\t"+str((pooled_singletons+ pooled_doubletons+pooled_tripletons+pooled_quadrupletons)/pooled_segregating)+"\n")
            simgenes_summary_stats_file.write("\nFraction of rare alleles (below "+str(rare_number_temp/sample_size_temp)+" percent) = "+"\t"+str((pooled_singletons+ pooled_doubletons+pooled_tripletons+pooled_quadrupletons)/pooled_segregating)+"\n")
        else:
            simgenes_summary_stats_file.write("fraction of singleton biallelic sites (genes with xbar=1/sample_size) = \tnan (0 segregating sites)\n")

            simgenes_summary_stats_file.write("fraction of doubleton biallelic sites (genes with xbar=2/sample_size) = \tnan (0 segregating sites)\n")

            simgenes_summary_stats_file.write("combined fraction of singleton and doubleton biallelic sites = \tnan (0 segregating sites)\n")

            simgenes_summary_stats_file.write("combined fraction of singleton, doubleton, and tripleton biallelic sites = \tnan (0 segregating sites)\n")
            simgenes_summary_stats_file.write("combined fraction of singleton-to-quadrupleton biallelic sites = \tnan (0 segregating sites)\n")
            
            simgenes_summary_stats_file.write("\nFraction of rare alleles (below "+str(rare_number_temp/sample_size_temp)+" percent) = \tnan (0 segregating sites)\n")
    
    





#    simgenes_summary_stats_file.write("xbar = "+"\t"+str(xbar)+"\n")
#    simgenes_summary_stats_file.write("x2bar = "+"\t"+str(x2bar)+"\n")
#    simgenes_summary_stats_file.write("pi = "+"\t"+str(pi)+"\n")
#    simgenes_summary_stats_file.write("segregating sites = "+"\t"+str(segregating_sites)+"\n")
#    simgenes_summary_stats_file.write("num singletons = "+"\t"+str(singleton_count_temp)+"\n")
#    simgenes_summary_stats_file.write("singletons/seg_sites = "+"\t"+str(singleton_count_temp/segregating_sites)+"\n")
#    simgenes_summary_stats_file.write("fixations = "+"\t"+str(fixed_mutations_temp)+"\n")


    simgenes_summary_stats_file.write("\n\n\n--- Total runtime: %s minutes ---\n" % ((time.time() - start_time)/60))
    

    simgenes_summary_stats_file.close()


    if print_SFS_over_sim_genes==1:
        os.chdir("../..")
    else:
        os.chdir("..")
#    os.chdir("..")







########################################################################################################
###################      ######   ###########    ##########       #####     ######    ..................
######################   ######   ########   ###  ###########   ########     #####  ####################
######################    ####    ######   #######   ########   ########   #  ####  ####################
######################  #  #  ##  ######   #######   ########   ########   ##  ###  ####################
######################  ##   ###  ######             ########   ########   ###  ##  ####################
######################  ########  ######   #######   ########   ########   ####  #  ####################
######################  ########  ######   #######   ########   ########   #####    ####################
#..................     ########   ....    #######    .....       ....     ######     ##################
####################################################################################################djb#



#______________________________________________________________
#______________________________________________________________
#
#                       MAIN FUNCTION
#______________________________________________________________
#______________________________________________________________




def main(initpopsize,L,mu,s, h, complete_underdominance, burnin_multiple, demography, ancestry, growth, sample_size, sample_data, filename, sparse_SFS, linearS,linearL,linearU, number_simulated_genes, simulated_gene_length, poisson_genes, recurrent_flag, recurrent_kernel, printrawSFS_flag, printsampleSFS_flag, printgenes_flag, printgenicSFS_flag, sim_genes_before_sampling, sim_genes_after_sampling, length_list_name, damaging_or_synon, rare_frac_definition, print_each_gene_SFS_flag, seedstart, russiandoll, seed):
    
    
    
    starting_directory = os.getcwd()

    burnin = int(burnin_multiple*initpopsize)
    demog_runtime = 5920+1
    
    #### Define growth rate for linear and exponential
    exponential_growth_rate=growth
    linear_growth_rate=growth
#    exponential_growth_rate = 0.001  ## FOR EXPONENTIAL MODEL
#    linear_growth_rate = 10  ## FOR LINEAR MODEL
    #### Define second growth phase for Tennessen model
    second_growth_init =18900 # just a temporary value.  dynamically stiched to previous exponential
    #### Define third growth phase for Browning model
    third_growth_init =753448 # just a temporary value.  dynamically stiched to previous exponential



    if filename is None:
        fout = sys.stdout
    else:
        fout = open(filename + ".out", "w")
    fout.write("\n\n ############# NEW SIMULATION RUN ############### \n\n")

    ##### This is if you want diploids from haploid input (default is haploid population size = 2N)
    if is_diploid==1:
        old_popsize = 2*initpopsize
        new_popsize = 2*initpopsize
    else:
        old_popsize = initpopsize
        new_popsize = initpopsize





    if s>0:
        fout.write("\n\n\t WARNING: S IS POSITIVE (BENEFICIAL MUTATIONS)  \n\n")

    fout.write("Demography type = "+ str(demography)+"\n")
    fout.write("Initial haploid population size (2N) = "+ str(initpopsize )+"\n")
#    fout.write("Log10 population size = "+ str(np.log10(initpopsize)    )+"\n")
    fout.write("Mutation rate per site = "+ str(mu)+"\n")
    fout.write("Number of sites = "+ str(L)+"\n")
    fout.write("Total mutation rate per individual = "+ str(mu*L)+"\n")
    fout.write("selection coefficient (negative is deleterious) = "+ str(s)+"\n")
    fout.write("Dominance coefficient = "+ str(h)+"\n")
    if complete_underdominance:
        fout.write("WARNING! Running complete underdominance with S_het = hs, S_hom = 0."+"\n")
    fout.write("Number of generations burnin = "+ str(burnin)+"\n")




    fixed_mutations = 0
    singleton_count=0
    counts = np.array([])




    #_______________________________________________________________
    #_______________________________________________________________
    #
    #                    EQUILIBRATION BURNIN
    #_______________________________________________________________
    #_______________________________________________________________

    fout.write( "Equilibrating initial population\n")
    for generation_burn in range(burnin):
        new_popsize, old_popsize = demography_equilibrium(old_popsize)
        counts, fixed_mutations = drift_step(s, h, complete_underdominance, new_popsize, old_popsize, counts, fixed_mutations, fout,0*generation_burn)

        if recurrent_flag:
            if recurrent_kernel=="v3":
                # Slow recurrent kernel that adds monomorphic sites before poisson sampling
                counts = mutation_step_recurrent_v3(mu,L, new_popsize, counts, fout)
            elif recurrent_kernel=="slow":
                # recurrent_v3 adds
                counts = mutation_step_recurrent_vSLOW(mu,L, new_popsize, counts, fout)
            elif recurrent_kernel=="multinomial":
                # Slow recurrent kernel that adds monomorphic sites before poisson sampling
                if generation_burn==0:
                    k_max = poisson_cutoff(mu, new_popsize,L)
                    pdf_truncated_poisson = truncated_poisson(mu, new_popsize,k_max, fout)
                elif generation_burn%(new_popsize/2)==0 and generation_burn>0:
                    if counts.size<L:
                        monomorphic_temp = (L-counts.size)
                        k_max = poisson_cutoff(mu, new_popsize,monomorphic_temp)
                        pdf_truncated_poisson = truncated_poisson(mu, new_popsize,k_max,fout)
                    else:
                        pdf_truncated_poisson=np.array([1,0])

                        
                        
                counts = mutation_step_recurrent_vMultinomial(mu,L, new_popsize, counts, pdf_truncated_poisson, generation_burn,fout)

            else:
                fout.write("\nERROR: CANNOT FIND SPECIFIED RECURRENT KERNEL!\n")
        else:
            counts = mutation_step(mu,L, new_popsize, counts, fout)
        old_popsize=new_popsize
        ####  Print generation, current and previous population size
        if generation_burn%(burnin/10)==0 or generation_burn == (burnin-1):
            fout.write("Burn generation:\t"+str(generation_burn)+",\tInitialized 2N:\t"+ str(new_popsize)+"\n")
            fout.write("\n--- Runtime thus far: %s minutes ---\n" % ((time.time() - start_time)/60))

    #_______________________________________________________________
    #_______________________________________________________________
    #
    #                          DEMOGRAPHY
    #_______________________________________________________________
    #_______________________________________________________________

    equilibrium_model = 0
    linear_growth = 0
    exponential_growth = 0
    tennessen_model = 0
    super_tennessen_model=0
    browning_model=0
    gaussian_model=0
#    keinan_model = 0


    if initpopsize!=28948:
        fout.write( "WARNING: Deviating from human ancestral population size!\n")
    if demography=="equilibrium":
        fout.write( "Running equilibrium demography...burnin only\n")
        equilibrium_model=1
    elif demography=="linear":
        fout.write( "Running linear growth demography\n")
        linear_growth=1
    elif demography=="exponential":
        fout.write( "Running exponential growth demography\n")
        exponential_growth=1
    elif demography=="tennessen":
        fout.write( "Running "+ancestry+" Tennessen demography\n")
        tennessen_model=1
    elif demography=="supertennessen":
        fout.write( "Running "+ancestry+" Tennessen demography with faster recent expansion\n")
        super_tennessen_model=1
    elif demography=="gaussian":
        fout.write( "Running "+ancestry+" Tennessen demography with Gaussian recent expansion\n")
        gaussian_model=1
    elif demography=="browning":
        fout.write( "Running "+ancestry+" Tennessen demography with third exponential phase (Browning model)\n")
        browning_model=1




    else:
        fout.write( "Demography not recognized....terminating simulation.\n")


    for generation in range(demog_runtime+1):

        if equilibrium_model==1:
            break;

        elif linear_growth==1:
            new_popsize = demography_lineargrowth(old_popsize, linear_growth_rate, generation)
        
        elif exponential_growth==1:
            new_popsize = demography_exponential(old_popsize, exponential_growth_rate, generation)

        elif tennessen_model==1:
            new_popsize = demography_tennessen(initpopsize,generation, ancestry, second_growth_init)
            if generation == 5715:
                second_growth_init =new_popsize

        elif super_tennessen_model==1:
#            if growth <= 0.0195 and generation==0:
#                fout.write("WARNING: Super-Tennessen being run with Tennessen rate.\n")
#                growth=0.0195
            new_popsize = demography_super_tennessen(initpopsize,generation, ancestry, second_growth_init, growth)
            if generation == 5715:
                second_growth_init =new_popsize

        elif gaussian_model==1:
            new_popsize = demography_gaussian_tennessen(initpopsize,generation, ancestry, second_growth_init, growth)
            if generation == 5715:
                second_growth_init =new_popsize


        elif browning_model==1:
            if growth==0.0195:
                IBD_growth = 0.0833
            else:
                IBD_growth = growth

            new_popsize = demography_browning(initpopsize,generation, ancestry, second_growth_init, third_growth_init, IBD_growth)
            if generation == 5715:
                second_growth_init =new_popsize
            if generation == 5902:
                third_growth_init =new_popsize



        else:
            fout.write("WARNING: No demography run.  Equilibration only.")
            break
        counts, fixed_mutations = drift_step(s, h, complete_underdominance, new_popsize, old_popsize, counts, fixed_mutations, fout, generation) #  drift based on frequency in last generation and new population size


        if recurrent_flag:
            if recurrent_kernel=="slow" or recurrent_kernel=="multinomial":
                # Slow recurrent kernel that adds monomorphic sites before poisson sampling
                counts = mutation_step_recurrent_vSLOW(mu,L, new_popsize, counts, fout)
            elif recurrent_kernel=="v3":
                # recurrent_v3 adds
                counts = mutation_step_recurrent_v3(mu,L, new_popsize, counts, fout)
            else:
                fout.write("\nERROR: CANNOT FIND SPECIFIED RECURRENT KERNEL!\n")
        else:
            counts = mutation_step(mu,L, new_popsize, counts, fout) #  mutate with recurrent mutations
        ####  Print generation, current and previous population size
        if generation%100==0 or generation == (demog_runtime-1):
            fout.write("Generation:\t"+str(generation)+",\tOld 2N:\t"+ str(new_popsize)+",\tNew 2N:\t"+ str(old_popsize)+"\n")
        old_popsize=new_popsize #    update population size


    if L<counts.size:
        fout.write("WARNING: NUMBER OF SEGREGATING SITES EXCEEDS LENGTH OF GENOME...trimming")
        counts=counts[:L]

    fout.write("Number of fixed mutations = "+ str(fixed_mutations)+"\n")

    fout.write("\n--- Runtime thus far: %s minutes ---\n" % ((time.time() - start_time)/60))

    #   Create raw SFS prior to downsampling





    #_______________________________________________________________
    #
    #    Printing raw SFS
    #_______________________________________________________________


    fout.write("\nMaking SFS file\n")


    if recurrent_flag:
        assure_path_exists("SFS_output_v"+simdose_version+"_recurrent")
        os.chdir("SFS_output_v"+simdose_version+"_recurrent")
    else:
        assure_path_exists("SFS_output_v"+simdose_version)
        os.chdir("SFS_output_v"+simdose_version)


    if demography=="tennessen":

        assure_path_exists("tennessen")
        os.chdir("tennessen")


    elif demography=="supertennessen":

        assure_path_exists("supertennessen")
        os.chdir("supertennessen")

    elif demography=="browning":
        
        assure_path_exists("browning")
        os.chdir("browning")

    elif demography=="linear":

        assure_path_exists("linear")
        os.chdir("linear")


    elif demography=="exponential":

        assure_path_exists("exponential")
        os.chdir("exponential")

    elif demography=="gaussian":

        assure_path_exists("gaussian_growth")
        os.chdir("gaussian_growth")

    elif demography=="equilibrium":

        assure_path_exists("equilibrium")
        os.chdir("equilibrium")


####  Make a new folder for complete underdominance.
####  Can instead alter filename by editing each output file below.
    if complete_underdominance:

        assure_path_exists("complete_underdominance")
        os.chdir("complete_underdominance")




    assure_path_exists("raw_SFS")
    os.chdir("raw_SFS")



    ignore_sample_size = 1
    final_popsize=new_popsize
    
    raw_SFS, singleton_count = print_raw_SFS(initpopsize, new_popsize, counts, fixed_mutations, demography, ancestry, s, h, mu, L, growth, sparse_SFS, linearS, linearL, linearU, "SFS", ignore_sample_size, seed, printrawSFS_flag, 0) # the last input is to not write sample size


    fout.write("\nMaking summary stats file\n\n")

    assure_path_exists("summary_stats")
    os.chdir("summary_stats")


    print_summary_stats(initpopsize, new_popsize, counts, fixed_mutations, demography, ancestry,s, h, complete_underdominance, mu,L, singleton_count, growth, linearS, linearL, linearU, "stats", ignore_sample_size, final_popsize, seed)  # the last input is to not write sample size

    #_______________________________________________________________
    #
    #    Simulate genes from raw counts
    #_______________________________________________________________


    #Define fraction of rare alleles cutoff
    rare_number=int(rare_frac_definition*new_popsize)+1


    if number_simulated_genes!=0 and sim_genes_before_sampling==1:
        
        assure_path_exists("../../simulated_genes_rawSFS_v2.3.4R")
        os.chdir("../../simulated_genes_rawSFS_v2.3.4R")

        if simulated_gene_length!=0:
            gene_length_list = np.array([simulated_gene_length])
        else:
            gene_length_list = [30, 100, 300, 1000, 3000, 10000, 30000]
#        gene_length_list = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
        for L_gene in gene_length_list:
            ##  Create summary stats xbar, pi, logstat, etc. for number_genes genes of length gene_length
            
            ## Simulate genes by Poisson sampling from sample SFS (may introduce some noise)
            if poisson_genes==1:
                fout.write("\nPoisson sampling rawSFS to simulate genes\n\n")
                gene_total_counts, gene_xbar, gene_x2bar, gene_pi, gene_logstat,gene_logstatUscale,gene_logstatMUscale,gene_logstatLscale,gene_logload,gene_logload2, gene_segregating_sites, gene_singletons,gene_frac_rare = simulate_genes_poisson(number_simulated_genes, L_gene, mu,raw_SFS, L, new_popsize, rare_number, fout)
#                assure_path_exists("../poisson_simulated_genes_v"+simdose_version)
#                os.chdir("../poisson_simulated_genes_v"+simdose_version)
                assure_path_exists("../poisson_simulated_genes")
                os.chdir("../poisson_simulated_genes")
            ## Simulate genes by randomly sampling sites
            else:
                fout.write("\nShuffling counts to simulate genes from rawSFS\n\n")
                gene_total_counts, gene_xbar, gene_x2bar, gene_pi, gene_logstat,gene_logstatUscale,gene_logstatMUscale,gene_logstatLscale,gene_logload,gene_logload2, gene_segregating_sites, gene_singletons,gene_frac_rare  = simulate_genes(number_simulated_genes, L_gene, mu,counts, L, new_popsize,rare_number, fout)
            fout.write("\n--- Runtime thus far: %s minutes ---\n" % ((time.time() - start_time)/60))
            fout.write("\nPrinting simulated genes (rawSFS)\n\n")
            print_simulated_genes(number_simulated_genes, L_gene, gene_total_counts, gene_xbar, gene_x2bar, gene_pi, gene_logstat,gene_logstatUscale,gene_logstatMUscale,gene_logstatLscale,gene_logload,gene_logload2, gene_segregating_sites, gene_singletons,gene_frac_rare , rare_number, initpopsize, demography, ancestry, s, h, mu, L, growth, new_popsize, "",linearS,linearL,linearU,"SimGenes", seed, final_popsize, printgenes_flag, printgenicSFS_flag, sparse_SFS,gene_length_list, length_list_name, print_each_gene_SFS_flag, geneSFS)
    
    




    
    
    
    #_______________________________________________________________
    #
    #    Downsample raw counts to sample size
    #_______________________________________________________________
    
    

    #   Downsample raw SFS to make population sample SFS
    if sample_size==0 and sample_data=="exac" and ancestry=="european":
        sample_size=68858 # number of hapoid individuals (chromosomes) in ExAC Europeans
    elif sample_size==0 and sample_data=="exac" and ancestry=="african":
        sample_size=10406 # number of hapoid individuals (chromosomes) in ExAC Africans

    fout.write("\nDownsampling to appropriate sample size\n")

    #  INSTEAD OF DOWNSAMPLING SFS, JUST BINOMIALLY SAMPLE ALLELES (neutrally drift one generation)
    sample_counts, fixed_mutations = drift_step(0, 0, 0, sample_size, new_popsize, counts, fixed_mutations, fout, 1234)  #  1234 is just a random number.  As long as this is not generation

    assure_path_exists("../../sampleSFS_size_"+str(sample_size))
    os.chdir("../../sampleSFS_size_"+str(sample_size))

    fout.write("\nMaking sample SFS file\n")
    sample_SFS, sample_singleton_count = print_raw_SFS(initpopsize, sample_size, sample_counts, fixed_mutations, demography, ancestry, s, h, mu, L, growth, sparse_SFS, linearS, linearL,linearU, "SFS",sample_size, seed, printsampleSFS_flag, 0)

    if russiandoll:
        # produce sample SFS files for each order of magnitude shorter than the total length
        if L>sample_counts.size:
            sample_counts_with_monomorphic=np.append(sample_counts,np.zeros(int(L-sample_counts.size)))
        else:
            sample_counts_with_monomorphic=sample_counts
        np.random.shuffle(sample_counts_with_monomorphic)

#        fout.write(str(np.log10(L)))
        nested_power = np.arange(1,np.floor(np.log10(L))-1)

        for Lpower in nested_power:
            L_russiandoll = int(L/(10**Lpower))
            there_are_no_fixed_mutations=0
            print_raw_SFS(initpopsize, sample_size, sample_counts_with_monomorphic[:L_russiandoll], there_are_no_fixed_mutations, demography, ancestry, s, h, mu, L_russiandoll, growth, sparse_SFS, linearS, linearL,linearU,"SFS",sample_size, seed, printsampleSFS_flag, 0)

    fout.write("\n\n sample SFS shape = "+str(np.shape(sample_SFS))+"\n\n")

    fout.write("\nMaking sample summary stats file\n\n")

    assure_path_exists("sample_summary_stats")
    os.chdir("sample_summary_stats")

    print_summary_stats(initpopsize, sample_size, sample_counts, fixed_mutations, demography, ancestry,s, h, complete_underdominance, mu,L, sample_singleton_count, growth, linearS,linearL,linearU, "stats", sample_size, final_popsize, seed)

    fout.write("\n--- Runtime thus far: %s minutes ---\n" % ((time.time() - start_time)/60))





    #_______________________________________________________________
    #
    #    Simulate genes from downsampled counts
    #_______________________________________________________________

    #Define fraction of rare alleles cutoff
    rare_number=int(rare_frac_definition*sample_size)+1


    if number_simulated_genes!=0 and sim_genes_after_sampling==1:
        if poisson_genes==1:
#            assure_path_exists("../../poisson_simulated_genes_v"+simdose_version)
#            os.chdir("../../poisson_simulated_genes_v"+simdose_version)
            assure_path_exists("../../poisson_simulated_genes")
            os.chdir("../../poisson_simulated_genes")
        else:
#            assure_path_exists("../../simulated_genes_v"+simdose_version)
#            os.chdir("../../simulated_genes_v"+simdose_version)
             assure_path_exists("../../simulated_genes")
             os.chdir("../../simulated_genes")
            
        if length_list_name=="XX":

            if simulated_gene_length!=0:
                gene_length_list = np.array([simulated_gene_length])
            else:
                gene_length_list = [30, 100, 300, 1000, 3000, 10000, 30000]
    #        gene_length_list = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
            for L_gene in gene_length_list:
                fout.write("\nSimulating "+str(number_simulated_genes)+" genes of length "+str(L_gene)+"\n\n")
                ##  Create summary stats xbar, pi, logstat, etc. for number_genes genes of length gene_length
                
                ## Simulate genes by Poisson sampling from sample SFS (may introduce some noise)
                if poisson_genes==1:
                    fout.write("\nPoisson sampling SFS to simulate genes\n\n")


                    if print_each_gene_SFS_flag==1:

                        gene_total_counts, gene_xbar, gene_x2bar, gene_pi, gene_logstat,gene_logstatUscale,gene_logstatMUscale,gene_logstatLscale,gene_logload,gene_logload2, gene_segregating_sites, gene_singletons,gene_frac_rare  = simulate_genes_poisson_with_SFSprint(number_simulated_genes, L_gene,mu, sample_SFS, L, sample_size, rare_number, initpopsize, demography,ancestry, s, h, growth, sparse_SFS,linearS,linearL,linearU,seedstart, fout)


                    else:
                        gene_total_counts, gene_xbar, gene_x2bar, gene_pi, gene_logstat,gene_logstatUscale,gene_logstatMUscale,gene_logstatLscale,gene_logload,gene_logload2, gene_segregating_sites, gene_singletons,gene_frac_rare  = simulate_genes_poisson(number_simulated_genes, L_gene,mu, sample_SFS, L, sample_size, rare_number, fout)

                ## Simulate genes by randomly sampling sites
                else:
                    fout.write("\nShuffling counts to simulate genes\n\n")
                    geneSFS, gene_total_counts, gene_xbar, gene_x2bar, gene_pi,  gene_logstat,gene_logstatUscale,gene_logstatMUscale,gene_logstatLscale,gene_logload,gene_logload2, gene_segregating_sites, gene_singletons,gene_frac_rare  = simulate_genes(number_simulated_genes, L_gene,mu, sample_counts, L, sample_size,rare_number, fout)
                fout.write("\n--- Runtime thus far: %s minutes ---\n" % ((time.time() - start_time)/60))
                fout.write("\nPrinting simulated genes\n\n")
                print_simulated_genes(number_simulated_genes, L_gene, gene_total_counts, gene_xbar, gene_x2bar, gene_pi, gene_logstat,gene_logstatUscale,gene_logstatMUscale,gene_logstatLscale,gene_logload,gene_logload2, gene_segregating_sites, gene_singletons,gene_frac_rare, rare_number ,initpopsize, demography, ancestry, s, h, mu, L, growth, sample_size, sample_data,linearS,linearL,linearU,"SimGenes", seed, final_popsize, printgenes_flag, printgenicSFS_flag, sparse_SFS, gene_length_list, length_list_name, print_each_gene_SFS_flag, geneSFS)

#                ####### Print per gene SFS
#                gene_length_list_temp = L_gene*np.ones(number_simulated_genes)
#                simulate_genes_fixed_list_with_SFSprint(gene_length_list_temp.astype(int),mu,number_simulated_genes, sample_counts, sample_SFS, L, sample_size,poisson_genes,rare_number, initpopsize, demography,ancestry, s, h, growth, sparse_SFS,linearS,linearL,linearU, fout)


        elif length_list_name!="XX":

#            os.chdir("../../_gene_length_lists")
            fixed_gene_length_list = import_gene_length_list(length_list_name,damaging_or_synon, fout)

            # randomly sample from gene length list
            gene_length_list = np.random.choice(fixed_gene_length_list,number_simulated_genes)
            fout.write("\nSimulating "+str(number_simulated_genes)+" genes sampled from length list "+length_list_name+"\n\n")
            if poisson_genes==1:
#                path =starting_directory+"/SFS_output_v"+simdose_version+"/"+demography+"/poisson_simulated_genes_v"+simdose_version
                path =starting_directory+"/SFS_output_v"+simdose_version+"/"+demography+"/poisson_simulated_genes"
                assure_path_exists(path)
                os.chdir(path)
            else:
#                path =starting_directory+"/SFS_output_v"+simdose_version+"/"+demography+"/simulated_genes_v"+simdose_version
                path =starting_directory+"/SFS_output_v"+simdose_version+"/"+demography+"/simulated_genes"
                os.chdir(path)
            ##  Create summary stats xbar, pi, logstat, etc. for number_genes genes of length gene_length
            if print_each_gene_SFS_flag==1:
            
                gene_total_counts, gene_xbar, gene_x2bar, gene_pi, gene_logstat,gene_logstatUscale,gene_logstatMUscale,gene_logstatLscale,gene_logload,gene_logload2, gene_segregating_sites, gene_singletons,gene_frac_rare  = simulate_genes_fixed_list_with_SFSprint(gene_length_list,mu,number_simulated_genes, sample_counts, sample_SFS, L, sample_size,poisson_genes,rare_number, initpopsize, demography,ancestry, s, h, growth, sparse_SFS,linearS,linearL,linearU, fout)
            
            
            else:
                gene_total_counts, gene_xbar, gene_x2bar, gene_pi, gene_logstat,gene_logstatUscale,gene_logstatMUscale,gene_logstatLscale,gene_logload,gene_logload2, gene_segregating_sites, gene_singletons,gene_frac_rare  = simulate_genes_fixed_list(gene_length_list,mu,number_simulated_genes, sample_counts, sample_SFS, L, sample_size,poisson_genes,rare_number, fout)
#            
#            fout.write("\n\n LList = "+str(gene_length_list[1:10])+" \n")
#            fout.write("\n\n gene_L_list = "+str(fixed_gene_length_list[1:10])+" \n")
#            fout.write("\n\n gene_xbar = "+str(gene_xbar[1:10])+" \n\n")
#
            fout.write("\n--- Runtime thus far: %s minutes ---\n" % ((time.time() - start_time)/60))
            fout.write("\nPrinting simulated genes\n\n")
            L_gene=1 #this number is irrelevant but needs to be passed to print function
            print_simulated_genes(number_simulated_genes, L_gene, gene_total_counts, gene_xbar, gene_x2bar, gene_pi, gene_logstat,gene_logstatUscale,gene_logstatMUscale,gene_logstatLscale,gene_logload,gene_logload2, gene_segregating_sites, gene_singletons,gene_frac_rare ,rare_number, initpopsize, demography, ancestry, s, h, mu, L, growth, sample_size, sample_data,linearS,linearL,linearU,"SimGenes", seed, final_popsize, printgenes_flag, printgenicSFS_flag, sparse_SFS, gene_length_list, length_list_name, print_each_gene_SFS_flag, geneSFS)




    fout.write("\n--- Total runtime: %s minutes ---\n" % ((time.time() - start_time)/60))
    fout.close()





#______________________________________________________________
#______________________________________________________________
#
#                   COMMAND LINE INPUTS
#______________________________________________________________
#______________________________________________________________



if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option("--mutation-rate", "-U",type="float", dest="mu", default=1e-8, help="mutation rate mu; default 1e-8")
    parser.add_option("-R","--recurrent", dest="recurrent",action="store_true", default=False, help="Mutational process includes recurrent mutations ; default=FALSE")
    parser.add_option("--Rkernel", type="str", default="slow", help="Specify recurrent mutation kernel version ; default=v3")
    parser.add_option("--selection", "-S", type="float", dest="s", default=-0.1, help="selection coefficient s; default -0.1")
    parser.add_option("--dominance", "-H", type="float", dest="h", default=0.5, help="dominance h; default 0.5")
    parser.add_option("--underdom", "--underdominant", dest="underdom", action="store_true", default=False, help="Apply complete underdominance (s_het = hs, s_hom = 0); default = FALSE")
    parser.add_option("--length", "-L", dest="length",type="int", default=10000, help="locus length; default 1000")
    parser.add_option("--demography", "-D", type="string", dest="demography",default="equilibrium", help="filename prefix to print output to; default prints to stdout")
    parser.add_option("--ancestry", "-A", dest="ancestry", type="string", default="european", help="Ancestry for realistic demography (e.g. Tennessen model); default is european")
    parser.add_option("--growthrate", "-G", type="float", dest="growth", default=0.0195, help="Growth rate for exponential and linear models; default 0.0195")
    parser.add_option("--initpopsize","-N", dest="initpopsize",type="int", default=28948, help="initial haploid (twice diploid) population size 2N; default=28948 (ancestral human population)")
    parser.add_option("--burnin","-T", dest="burnin",type="float", default=5, help="multiple of 2N for number of generations; default=1000")
    parser.add_option("--notsparse", dest="sparse", action="store_false", default=True, help="Print full SFS instead of sparse SFS ; default=FALSE (sparse SFS will be printed)")
    ####  Options for downsampling to population sample size
    parser.add_option("--samplesize", type="int", default=100, help="Sample size for downsampled SFS ; default=100")
    parser.add_option("--data", type="string", default="", help="Source of sample data.  Used to determine sample size; default is exac")
    ####  Options for making simulated genes
    parser.add_option("--Ngenes", type="int", default=0, help="Number of simulated genes ; default=0 (off)")
    parser.add_option("--Lgenes", type="int", default=0, help="Length of simulated genes ; default=0 (use predetermined array)" )
    parser.add_option("--poissongenes", type="int", default=0, help="Poisson sample to simulate genes ; default=0")
    parser.add_option("--printrawSFS", type="int", default=1, help="Print raw SFS before downsampling ; default=1")
    parser.add_option("--printsampleSFS", type="int", default=1, help="Print SFS after downsampling ; default=1")
    parser.add_option("--printgeneSFS", type="int", default=0, help="Print SFS for each gene in different file (after downsampling) ; default=0")
    parser.add_option("--printgenestats", type="int", default=1, help="Print summary stats for individual simulated genes ; default=1")
    parser.add_option("--printbiallelicSFS", type="int", default=0, help="Print genic SFS of pooled sites for simulated genes ; default=0")
    
    parser.add_option("--simgenes_raw", type="int", default=0, help="Make simulated genes from raw SFS (before downsampling) ; default=0")
    parser.add_option("--simgenes_sample", type="int", default=1, help="Make simulated genes from sample SFS (after downsampling) ; default=1")
    parser.add_option("--lengthlist", type="str", default="XX", help="Gene length list to import ; default=XX")
    parser.add_option("--damagingorsynon", type="str", default="damaging", help="Import damaging or synon gene length list? ; default=damaging")

    parser.add_option("--fracrare", type="float", default=0.01, help="Define frequency cutoff for fraction of rare; default 0.01 (rare <= 1 percent)")

    parser.add_option("--geneSFS_startnumber", type="int", default=0, help="Starting point for gene number label for geneSFS and genesummary; default=0")
    parser.add_option("--russiandoll", action="store_true", default=False, help="Produce a single gene of various nested lengths from a larger length simulation ;default = FALSE")
    ####  Filename specification options
    parser.add_option("--fileprefix", "-f", type="string", dest="filename", default=None, help="filename prefix to print output to; default prints to stdout")
    parser.add_option("--run", type="str", dest="seed", default="XX", help="seed prefix for parallelization ; default=XX")
    parser.add_option("--linearL", action="store_true", default=False, help="Name files in linear L?; default = FALSE")
    parser.add_option("--linearU", action="store_true", default=False, help="Name files in linear U?; default = FALSE")
    parser.add_option("--linearS", action="store_true", default=False, help="Name files in linear s? ; default=0")

    opts, args = parser.parse_args()
    


    main(opts.initpopsize,opts.length,opts.mu, opts.s, opts.h, opts.underdom, opts.burnin, opts.demography, opts.ancestry, opts.growth, opts.samplesize, opts.data, opts.filename, opts.sparse, opts.linearS, opts.linearL, opts.linearU, opts.Ngenes, opts.Lgenes, opts.poissongenes, opts.recurrent, opts.Rkernel, opts.printrawSFS, opts.printsampleSFS, opts.printgenestats, opts.printbiallelicSFS, opts.simgenes_raw, opts.simgenes_sample, opts.lengthlist, opts.damagingorsynon, opts.fracrare, opts.printgeneSFS, opts.geneSFS_startnumber, opts.russiandoll, opts.seed)


