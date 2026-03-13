# Physics-Informed LSTM surrogate model for real-time street-scale flood forecasting 

Physics-Informed LSTM (PI LSTM) surrogate model is developed for street-scale flood forecasting, treating each street as a control volume for a complex coastal-urban system. 
This is done by adding the mass balance (MB) equation into the customized loss function of the LSTM surrogate model. This script is implemented in Keras Python.

Here, the total loss (L_total ) combines the data loss (L_data) with the physics-informed loss (L_phy) as follows:

<!--
L_total= α * L_data+ β * L_phy	
-->

$$
L_{total} = \alpha L_{data} + \beta L_{phy}
$$


**Data Loss**

Data loss $L_{data}$ is formulated to ensure that the predicted flood depths and volumes are consistent with ground-truth flood depths and volumes
$(h = \hat{h})$ and $(v = \hat{v})$.

<!--
$$
L_{data} =
\frac{1}{N}\sum_{n=1}^{N}(h_t^n - \hat{h}_t^n)^2 +
\frac{1}{N}\sum_{n=1}^{N}(v_t^n - \hat{v}_t^n)^2
$$
-->

**Physics Loss**

Physics loss $L_{phy}$ is formulated in two ways:

  **1. Exact Equation**
  The change of flood volume at the current timestep is equal to the net difference between the total inflow and the total outflow that occurred during the current and previous timestep.

$$
(\hat{v}_t^n - \hat{v}_{t-1}^n) - (Qin_{(t-1,t]}^n \Delta t + R_{(t-1,t]}^n \cdot Street\ Area - Qout_{(t-1,t]}^n \Delta t - Qpipe_{(t-1,t]}^n \Delta t) = 0
$$

  **2. Inequality Equation**
  The change of flood volume at the current timestep can not exceed the flow entering the system between the current and previous timestep.

$$
(\hat{v}_t^n - v_{t-1}^n) -
(Qin_{(t-1,t]}^n \Delta t + R_{(t-1,t]}^n \cdot Street\ Area) \le 0
$$

$$
(v_{t+1}^n - \hat{v}_t^n) -
(Qin_{(t,t+1]}^n \Delta t + R_{(t,t+1]}^n \cdot Street\ Area) \le 0
$$


The input features used for the data loss are hourly rainfall, hourly tide level, elevation, Topographic Wetness Index (TWI), and Depth-to-Water (DTW). The input features used for the physics loss are hourly inflow volume, hourly outflow volume, hourly rainfall volume, and hourly pipe flow volume. The models includes two target features – flood depth and flood volume. The water depth raster is collected from the TUFLOW model through a coupled 1D/2D simulation for each hour throughout all storm events. The water volume is calculated by summing the hourly water depths over the inundated area within the street segment using the zonal statistics tool of ArcGIS Pro. The input data is available on Hydroshare (Roy, 2026).

The framework for PI LSTM surrogate model is shown in Figure 1.

<img width="975" height="640" alt="image" src="https://github.com/user-attachments/assets/cdcc3e7a-dad5-48fe-ba41-39216af444b5" />
Figure 1: General framework of PI LSTM model with mass balance equations

<br><br>

**References**

Roy, B. (2026). Input Data for A Physics-Informed LSTM Surrogate Model for Real-time Flood Forecasting at the Street-scale, HydroShare, http://www.hydroshare.org/resource/5748edbcff794649bf384d6dd807b7bd
