# **Checkers**

### *COLLABORATORS*
* Ethan Ford
* Nathan Freestone

### *Prerequisites and Info*
* We forked the checkers behavior from a git repository called ```gym-checkers```. We've added a new python file as well as modifications to the ```alpha_beta.py``` for this particular project.
* *Installations*
    * stable_baselines
    * os
    * time
    * numpy
    * MLPRegressor

## *Introduction*
* For our final project, we decided to train a game of checkers.  The ```alpha_beta.py``` file uses a search tree to determine the next move a player should take; the program lets us adjust the search depth. Using the model ```MLPRegressor```, we iterated a maximum of 100, and gained training data from a collection of 100 games. We then iterated that process 10 times to ensure as accurate of data as possible. To implement optimal stop, we calculated the average reward from 1 to 10 depths, while also implementing a time threshold of 10 seconds, so as to not waste time calculating reward for depths that require lots of computational time.

## *Analysis*
* After implementing our optimal stop algorithm, we wanted to visualize what how fast the search depths were computed. The following graph is Search Depth over Time with a time threshold of 10 seconds:
  ![Screenshot 2024-12-05 at 8 29 46 PM](https://github.com/user-attachments/assets/d076194e-c873-4585-89d5-ef9d42dcc9ef)
  <img width="920" alt="Screenshot 2024-12-06 at 12 19 52 PM" src="https://github.com/user-attachments/assets/e2427842-1a19-413f-bf14-b001c0f4dbd3">
* Based on our output, it looks like after a depth of four, computational time increases exponentially into the shape of a log function. This makes sense, as search farther down the tree will cost more time. If we look at the rewards output, we can see that the optimal depth that yielded the most reward was at a depth of four. We now know how far we need to search if we are restricted to a time limit of 10 seconds, however, what happens if we increase the time?
  ### *40s*
  <img width="956" alt="Screenshot 2024-12-06 at 12 26 20 PM" src="https://github.com/user-attachments/assets/502cc369-2ab2-422e-906d-ae91348ec8f4">
  <img width="1074" alt="Screenshot 2024-12-06 at 12 27 09 PM" src="https://github.com/user-attachments/assets/f0767f51-aada-4099-8d48-c951f65cfde5">

  ### *60s*
  <img width="980" alt="Screenshot 2024-12-06 at 12 30 33 PM" src="https://github.com/user-attachments/assets/c8b84a10-5f6d-407b-9a4d-ea674c4dbf04">
  <img width="1087" alt="Screenshot 2024-12-06 at 12 31 10 PM" src="https://github.com/user-attachments/assets/cc32eb0f-7f8f-4d00-af72-f25efef413bc">

  ### *90s*
  <img width="963" alt="Screenshot 2024-12-06 at 12 36 10 PM" src="https://github.com/user-attachments/assets/60af1e28-a3a2-4306-a04c-344e1c35012c">
  <img width="1086" alt="Screenshot 2024-12-06 at 12 36 55 PM" src="https://github.com/user-attachments/assets/dee019c1-3818-4c74-a82f-3b2937b9bfd5">

* From our graphs, we can deduce that depending on our time constraint, optimal depth varies. However, since the average checkers turn is around 30 second to a minute, it's safe to assume that we'll always search at a depth of about 4 or 5.
* Now that we know what our optimal depth would realistically be, how does solution quality vary among different depths? In order to do this, we created an arbitrary game state and evaluated the reward of it's next states recursively under the guise of a time limit:
  
### *PROCESS*
* **Iterate Through Depths**:
  For each depth from 1 to max_depth, the function evaluates the given game state using the ```evaluate_state_with_timer``` function, which ensures the evaluation respects the time constraint.

* **Calculate Rewards**:
  The reward for the game state is calculated by searching all possible moves up to the current depth.

* **Store Results**:
  At each depth, the computed reward (solution quality) is stored in the solution_qualities list.

* **Return Values**:
  The function returns a list of normalized solution qualities for visualization or further analysis.

Below are a couple graphs depicting different runs on solution quality vs. depth:
  <img width="983" alt="Screenshot 2024-12-06 at 1 02 59 PM" src="https://github.com/user-attachments/assets/34bf970c-73e8-4c86-8ff5-50522f59276f">
  <img width="976" alt="Screenshot 2024-12-06 at 1 04 45 PM" src="https://github.com/user-attachments/assets/bc92c13a-6674-4d2e-88fc-15afb5d330c0">
  <img width="977" alt="Screenshot 2024-12-06 at 1 06 41 PM" src="https://github.com/user-attachments/assets/2806c217-0328-4f2d-8c22-cf5934e72a43">

* According to the graphs, they seem to be a bit inconsistent starting from a depth of about 7. However, what's more important is the increasing relationship between solution quality and depth. As we search farther, we get better solutions, except for certain situations where our quality goes down at max depth.







