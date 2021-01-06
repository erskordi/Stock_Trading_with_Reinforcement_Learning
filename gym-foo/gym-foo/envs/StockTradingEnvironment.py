''' 
This is an example of creating an environment for stock trading 
    
observation_space contains all of the input variables we want our agent to consider before making, or not making a trade. 
In this example, we want our agent to “see” the stock data points (open price, high, low, close, and daily volume) for the last five days,
as well a couple other data points like its account balance, current stock positions, and current profit.

action_space will consist of three possibilities: buy a stock, sell a stock, or do nothing.
But this isn’t enough; we need to know the amount of a given stock to buy or sell each time. 
Using gym’s Box space, we can create an action space that has a discrete number of action types (buy, sell, and hold), 
as well as a continuous spectrum of amounts to buy/sell (0-100% of the account balance/position size respectively).

The last thing to consider before implementing our environment is the reward. 
We want to incentivize profit that is sustained over long periods of time. 
At each step, we will set the reward to the account balance multiplied by some fraction of the number of time steps so far.
'''

import gym
from gym import spaces
import numpy as np
import random

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

class StockTradingEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):

        '''
        In the constructor, we first define the type and shape of our action_space, which will contain all of the actions possible for an agent to take in the environment. 
        Similarly, we’ll define the observation_space, which contains all of the environment’s data to be observed by the agent.
        '''

        super(StockTradingEnvironment, self).__init__()
        self.df = df # pandas dataframe containing data for stocks 
        self.reward_range = (0, MAX_ACCOUNT_BALANCE) # reward_range is keyword

        #Define action/observation spaces. They must by gym.spaces objects
        # Actions of the format Buy x%, Sell x%,Hold, etc.
        self.action_space = spaces.Box(low=np.array([0,0]), high=np.array([3,1]), dtype=np.float16)

        # Prices contains open price, high value, low value, close value and daily volume
        self.observation_space = spaces.Box(low=0,high=1,shape=(6,6),dtype=np.float16)
    

    def reset(self):
        #Reset the environment to an initial position

        '''
        reset method will be called to periodically reset the environment to an initial state.
        This is followed by many steps through the environment, in which an action will be provided by the model and must be executed, and the next observation returned. 
        This is also where rewards are calculated
        '''
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the dataframe
        self.current_step = random.randint(0, len(self.df.loc[:,'Open'].values) - 6) # set the current step to a random point within the data frame for unique experiences

        #The _next_observation method compiles the stock data for the last five time steps, appends the agent’s account information, and scales all the values to between 0 and 1.

        return self._next_observation()

    def _next_observation(self):
        # Get the data points for the last 5 days and scale 0-1

        frame = np.array({self.df.loc[self.current_step:self.current_step+5,'Open'].values / MAX_SHARE_PRICE,
                          self.df.loc[self.current_step:self.current_step+5,'High'].values / MAX_SHARE_PRICE,
                          self.df.loc[self.current_step:self.current_step+5,'Low'].values / MAX_SHARE_PRICE,
                          self.df.loc[self.current_step:self.current_step+5,'Close'].values / MAX_SHARE_PRICE,
                          self.df.loc[self.current_step:self.current_step+5,'Volume'].values / MAX_SHARE_PRICE,})
        
        # Append additional data and scale each value between 0-1

        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES*MAX_SHARE_PRICE),
        ]], axis=0)

        return obs
    
    #Follows the step function that returns observation, reward, done, info
    def step(self, action):
        #Execute one time-step within the environment
        self._take_action(action) # execute action

        self.current_step += 1  # and move one step

        if self.current_step > len(self.df.loc[:,'Open'].values) - 6:
            self.current_step = 0
        
        delay_modifier = (self.current_step / MAX_STEPS ) # make the agent "patient"

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        # Set the current price to a random price within the timestep
        current_price = random.uniform(self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount of balance in shares
            total_possible = self.balance / current_price
            shares_bought = total_possible * amount
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought
        elif action_type < 2:
            # Sell amount of shares held
            shares_sold = self.shares_held * amount
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price
        
        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0


        

    def render(self, mode='human', close=False):
        #Render the environment on the screen

        '''
        render method may be called periodically to print a rendition of the environment. 
        This could be as simple as a print statement, or as complicated as rendering a 3D environment using openGL.
        '''
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis}')
        print(f'Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth}')
        print(f'Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
