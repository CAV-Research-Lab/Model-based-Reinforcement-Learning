### To better understand

```python mppi_fast.py``` for pendulum environment
How to run

``` python spacerobot_floating_base_mbrl.py``` # To run the floating base model based reinforcement learning.

### To evaluate the results

```python validate_weights.py```

```python run_saved.py``` # for visualization

The actions are saved while running the ```spacerobot_floating_base_mbrl.py```. Give the same name to load while running ```run_saved.py```

The trained weights are also saved while running ```spacerobot_floating_base_mbrl.py```.

A example of solved environment is available in the ```results``` folder. This folder contains trained_weights and the actions got from those  trained_weights


### A short description of the MBRL
A good idea of what is happening in space dynamics is obtained from [paper 1](https://surreyac.sharepoint.com/sites/ConnectedandAutonomousVehiclesGroup/PDRAs/Forms/AllItems.aspx?e=5%3Ad5deacbaa21e4964b84df122b97ac6c7&at=9&CT=1564652485537&OR=OWA%2DNT&CID=7f934271%2D83d4%2D4a11%2Dab49%2Daa1c04813f15&FolderCTID=0x012000388917293171FB46BCCDC1F5772A6C16&id=%2Fsites%2FConnectedandAutonomousVehiclesGroup%2FPDRAs%2FAshith%20Rajendra%20Babu%2FLibrary%2FReselved%20Motion%20Rate%20Control%20of%20Space%20Manipulators%20with%20Generalized%20Jacobian%20Matrix%2Epdf&parent=%2Fsites%2FConnectedandAutonomousVehiclesGroup%2FPDRAs%2FAshith%20Rajendra%20Babu%2FLibrary).

The library folder in CAV sharepoint gives several papers as well to understand the dynamics

Inside the sharepoint folder ```MPC_RL```, read the papers POLO, relation between MPC and RL, neural network dynamics of MBDRL..., handful of trials  to understand the concept

The algorithm2 defined in [paper 2](https://surreyac.sharepoint.com/sites/ConnectedandAutonomousVehiclesGroup/PDRAs/Forms/AllItems.aspx?e=5%3Ad5deacbaa21e4964b84df122b97ac6c7&at=9&CT=1564652485537&OR=OWA%2DNT&CID=7f934271%2D83d4%2D4a11%2Dab49%2Daa1c04813f15&FolderCTID=0x012000388917293171FB46BCCDC1F5772A6C16&id=%2Fsites%2FConnectedandAutonomousVehiclesGroup%2FPDRAs%2FAshith%20Rajendra%20Babu%2FLibrary%2FMPC%20n%20RL%2Fpath%20integral%20control%2FInformation%20Theoretic%20MPC%20for%20Model%2DBased%20Reinforcement%20Learning%2Epdf&parent=%2Fsites%2FConnectedandAutonomousVehiclesGroup%2FPDRAs%2FAshith%20Rajendra%20Babu%2FLibrary%2FMPC%20n%20RL%2Fpath%20integral%20control) is coded for getting the MBRL work.

Mode free RL gives optimal solutions compared to model based RL but it requires retraining when the reward chagnes. Model based RL, once the dynamics is trained, is reward independent and is capable of using with any reward function. MBRL becomes difficult when the state space is large.

Space env poses unknown challenges so the dynamics need to be updated at a predefined frequency so as to cater to the unmodelled things (a simulator like Mujoco will assume perfect conditions). 

env: spacerobot_env
