import random
import numpy as np

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit


def print_card(card):
    print(card.rank, card.suit)

def print_deck(delt):
    for i in delt:
        print_card(i)

class Game:
    def __init__(self, casino_hidden, casino_shown, player_hidden, player_shown,delt):
        self.casino_hidden=casino_hidden
        self.casino_shown=casino_shown
        self.player_hidden=player_hidden
        self.player_shown=player_shown
        self.delt=delt

    def create_duplicate(self):
        return Game(self.casino_hidden, self.casino_shown, self.player_hidden, self.player_shown, self.delt)
        
    def print_data_structure(self):
        print("casino_hidden:",) 
        print_card(self.casino_hidden)
        print("casino_shown:",) 
        print_deck(self.casino_shown)
        print("player_hidden:",) 
        print_card(self.player_hidden)
        print("player_shown:",) 
        print_deck(self.player_shown)
        print("delt:",)
        print(self.delt)


# creates new card given a "deck" matrix
def freshcard(delt_mat):
    x=random.randrange(1, 13)
    y=random.randrange(1, 4)

    if delt_mat[x][y]==1:
        freshcard(delt_mat)

    delt_mat[x][y]=1

    return (Card(x,y),delt_mat)


# deals two hands for blackjack
def gen_hands():
    delt=np.zeros((13, 4))

    card=freshcard(delt)
    casino_hidden=card[0]
    delt=card[1]

    card=freshcard(delt)
    casinoShow=card[0]
    delt=card[1]

    card=freshcard(delt)
    playerShow=card[0]
    delt=card[1]

    card=freshcard(delt)
    playerHidden=card[0]
    delt=card[1]

    return Game(casino_hidden,[casinoShow],playerHidden,[playerShow],delt)



#optimizes count for a given blackjack hand
def best_hand(hand): 
    count=0
    aces=0
    for card in hand:
        if card.rank==1:
            count+=11
            aces+=1
        elif card.rank>10:
            count+=10
        else:  
            count+=card.rank

    while count>21 and aces!=0:
        aces-=1
        count-=10
    return count



# "proven" textbook strategy on hit/stay for a starting blackjack hand
def textbook_strat(state):
    oppcard=state.player_shown[0].rank

    while True:
        casino_pos=best_hand(state.casino_shown+[state.casino_hidden])

        if casino_pos>16:
            return state
        if casino_pos>12 and oppcard<7:
            return state
        if casino_pos>11 and 3<oppcard<7:
            return state
        
        delt=state.delt
        card=freshcard(delt)
        draw=card[0]
        delt=card[1]
        state.casino_shown=state.casino_shown+[draw]
        state.delt=delt


#given a grid of initally random distributions for opponent cards/ own card
# follow ML strategy
def ML_strat(state,grid):
    opp_pos=best_hand(state.casino_shown)
    player_pos=best_hand(state.player_shown+[state.player_hidden])

    choice=grid[player_pos][opp_pos]

    while choice > 0:

        delt=state.delt
        card=freshcard(delt)
        draw=card[0]
        delt=card[1]

        state.player_shown=state.player_shown+[draw]
        state.delt=delt
        player_pos=best_hand(state.player_shown+[state.player_hidden])
        choice=grid[player_pos][opp_pos]

        #this cutoff (never hit on perfect) is necessary for initial randomization grid
        if player_pos>21: 
           break

    return state

# generate random grid for reinforcement learning
grid = np.random.random((35, 35))

# 0 for tie, 1 for ML win, -1 for proof win
def pick_winner(state):
    player=best_hand(state.player_shown+[state.player_hidden])
    casino=best_hand(state.casino_shown+[state.casino_hidden])

    if player==casino:
        return 0

    if player>21 and casino>21:
        if player>casino:
            return -1
        return 1
    
    if player>21:
        return -1
    if casino>21:
        return 1
    
    if player>casino:
        return 1
    else:
        return -1

# reinforcement learning, gives 1 if "good hit" (changes win to loss),
#  gives -1 if "bad hit" (changes loss to win) 

def good_hit(state):
    if len(state.player_shown)==1:
        return 0

    rmv_last_hit = state.create_duplicate()
    rmv_last_hit.player_shown=rmv_last_hit.player_shown[:-1]
    post=pick_winner(state)
    pre=pick_winner(rmv_last_hit)

    if post==1 and pre==-1:
        return 1
    elif post==-1 and pre==1:
        return -1
    else:
        return 0
    


# for x tests, simulates games and updates global matrix "grid" to follow
# optimal strategy
def learn(x):
    i=0
    while i<x:
        a=gen_hands()
        b=textbook_strat(a)
        c=ML_strat(b,grid)
        d=good_hit(c)
        
        # based on outcomes of hitting in state c, update grid 
        c.player_shown=c.player_shown[:-1]
        player=best_hand(c.player_shown+[c.player_hidden])
        casino=best_hand(c.casino_shown)

        if d==1:
            grid[player][casino]=grid[player][casino]+.1

        if d==-1:
            grid[player][casino]=grid[player][casino]-.1

        i+=1



# after grid is trained, test win rate for x tests  
def tester(x):
    winner=0
    ties=0
    i=0

    while i<x:
        a=gen_hands()
        b=textbook_strat(a)
        c=ML_strat(b,grid)

        if pick_winner(c)==1:
            winner+=1
        elif pick_winner(c)==0:
            ties+=1
        i+=1

    return winner/(x-ties) # note subtract out ties from overall win rate

print("simulating")
print(learn(10000000))
print("testing win rate % vs optimal strategy:")
print(tester(100000))


# proof based vs never hit-> %29.2 win rate for never hit

# proof based vs reinforcement learning model -> %54.23873703843507 win rate on 10000000 test cases
