import numpy as np
import drawHeatMap as hm
import rewardTable as rt
import transitionTable as tt

def expect(xDistribution, function):
    expectation=sum([function(x)*px for x, px in xDistribution.items()])
    return expectation

def getSPrimeRDistributionFull(s, action, transitionTable, rewardTable):
    reward=lambda sPrime: rewardTable[s][action][sPrime]
    p=lambda sPrime: transitionTable[s][action][sPrime]
    sPrimeRDistribution={(sPrime, reward(sPrime)): p(sPrime) for sPrime in transitionTable[s][action].keys()}
    return sPrimeRDistribution
    
def updateQFull(s, a, Q, getSPrimeRDistribution, gamma):

    Qas = 0 # Create empty variable to hold the result
    sprime_dist = getSPrimeRDistribution(s, a) # Generate the s prime distribution 

    for sprime_r, outcome_probability in sprime_dist.items(): # Loop through the s prime distribution and sum the values to get the new Q
        sprime = sprime_r[0]
        reward = sprime_r[1]
        Qas += outcome_probability * (reward+ gamma* max(Q[sprime].values()))

    return Qas
    

def qValueIteration(Q, updateQ, stateSpace, actionSpace, convergenceTolerance):

    Qnew = Q # Create new variable to hold the updated Q

    # Loop through states and actions to iterate through Q's
    for state in stateSpace:
        for action in actionSpace:
            newvalue = updateQ(state, action, Q)
            oldvalue = Q[state][action]
            while(abs(newvalue-oldvalue)>convergenceTolerance): # Check if the new value is within the convergence tolerance
                oldvalue = newvalue
                newvalue = updateQ(state, action, Q) # Update the new value with updateQ function
            Qnew[state][action] = newvalue # Store the new value in the new Q dictionary

    return Qnew

def getPolicyFull(Q, roundingTolerance):
    
    policy = {} # Create empty policy dictionary 
    max_Q = max(Q.values()) # Find the max from the current actions
    n_chosen_actions = 0 # Create counter for number of actions in the policy

    ## Count the number of actions within rounding tolerance of max

    for action_value in Q.items():
        Q_value = action_value[1]
        if (max_Q - Q_value < roundingTolerance):
            n_chosen_actions += 1


    ## Write policy for each of the chosen actions

    for action_value in Q.items():
        Q_value = action_value[1]
        action = action_value[0]
        if(max_Q - Q_value < roundingTolerance):
            policy[action] = 1/n_chosen_actions


    return policy


def viewDictionaryStructure(d, levels, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(levels[indent]) + ": "+ str(key))
        if isinstance(value, dict):
            viewDictionaryStructure(value, levels, indent+1)
        else:
            print('\t' * (indent+1) + str(levels[indent+1])+ ": " + str(value))

def main():
    
    minX, maxX, minY, maxY=(0, 3, 0, 2)
    
    actionSpace=[(0,1), (0,-1), (1,0), (-1,0)]
    stateSpace=[(i,j) for i in range(maxX+1) for j in range(maxY+1) if (i, j) != (1, 1)]
    Q={s:{a: 0 for a in actionSpace} for s in stateSpace}
    
    normalCost=-0.04
    trapDict={(3,1):-1}
    bonusDict={(3,0):1}
    blockList=[(1,1)]
    
    p=0.8
    transitionProbability={'forward':p, 'left':(1-p)/2, 'right':(1-p)/2, 'back':0}
    transitionProbability={move: p for move, p in transitionProbability.items() if transitionProbability[move]!=0}
    
    transitionTable=tt.createTransitionTable(minX, minY, maxX, maxY, trapDict, bonusDict, blockList, actionSpace, transitionProbability)

    rewardTable=rt.createRewardTable(transitionTable, normalCost, trapDict, bonusDict)

    """
    levelsReward  = ["state", "action", "next state", "reward"]
    levelsTransition  = ["state", "action", "next state", "probability"]
    
    viewDictionaryStructure(transitionTable, levelsTransition)
    viewDictionaryStructure(rewardTable, levelsReward)
    """
        
    getSPrimeRDistribution=lambda s, action: getSPrimeRDistributionFull(s, action, transitionTable, rewardTable)
    gamma = 0.8       
    updateQ=lambda s, a, Q: updateQFull(s, a, Q, getSPrimeRDistribution, gamma)
    
    convergenceTolerance = 10e-7
    QNew=qValueIteration(Q, updateQ, stateSpace, actionSpace, convergenceTolerance)
    
    roundingTolerance= 10e-7
    getPolicy=lambda Q: getPolicyFull(Q, roundingTolerance)
    policy={s:getPolicy(QNew[s]) for s in stateSpace}
    
    V={s: max(QNew[s].values()) for s in stateSpace}
    
    VDrawing=V.copy()
    VDrawing[(1, 1)]=0
    VDrawing={k: v for k, v in sorted(VDrawing.items(), key=lambda item: item[0])}
    policyDrawing=policy.copy()
    policyDrawing[(1, 1)]={(1, 0): 1.0}
    policyDrawing={k: v for k, v in sorted(policyDrawing.items(), key=lambda item: item[0])}

    hm.drawFinalMap(VDrawing, policyDrawing, trapDict, bonusDict, blockList, normalCost)

    
    
if __name__=='__main__': 
    main()
