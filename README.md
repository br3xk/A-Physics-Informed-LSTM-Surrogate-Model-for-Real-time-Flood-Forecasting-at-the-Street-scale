Physics-Informed LSTM (PI LSTM) surrogate model was developed for street-scale flood forecasting, treating each street as a control volume, in a complex coastal-urban system. 
The main research question of this study is: Does embedding physical laws of flood routing into an LSTM model improve the prediction skill of street-scale flood forecasting? 
This is investigated by adding the mass balance (MB) equation into the customized loss function of the LSTM surrogate model as below: 

L_total= α * L_data+ β * L_phy	

Here, the total loss (L_total ) combines the data loss (L_data), which ensures to match flood depths (h=(h_  ) ̂) and volumes (v=(v_  ) ̂), with the physics-informed loss (L_phy), which constrains the predictions to be physically plausible, that means the change of flood volume maintains the MB law. 

Data Loss

L_data=1/N ∑_(n=1)^N▒(h_t^n-〖h ̂_t^n〗_  )^2   + 1/N ∑_(n=1)^N▒(v_t^n-v ̂_t^n )^2 

Physics Loss
We used two formulations for physics loss. 
1. Exact Equation
The change of flood volume at the current timestep is equal to the net difference between the total inflow and the total outflow that occurred between the current and previous timestep.
(v ̂_t^n- v ̂_(t-1)^n)-(〖Qin〗_(t-1,t]^n*∆t+R_((t-1,t])^n*Street Area-〖Qout〗_(t-1,t]^n*∆t-〖Qpipe〗_(t-1,t]^n*∆t) = 0
2. Inequality Equation
The change of flood volume at the current timestep can not exceed the flow entering the system between the current and previous timestep.
(v ̂_t^n- v_(t-1)^n )-(〖Qin〗_(t-1,t]^n*∆t+R_((t-1,t])^n*Street Area)≤  0
(v_(t+1)^n- v ̂_t^n )-(〖Qin〗_(t,t+1]^n*∆t+R_((t,t+1])^n*Street Area)≤ 0

The input features used for the data loss were hourly rainfall, hourly tide level, elevation, Topographic Wetness Index (TWI), and Depth-to-Water (DTW). The input features used for the physics loss were hourly inflow volume, hourly outflow volume, hourly rainfall volume, and hourly pipe flow volume. The models included two target features – flood depth and flood volume. The water depth raster was collected from the TUFLOW model through a coupled 1D/2D simulation for each hour throughout all storm events. The water volume was calculated by summing the hourly water depths over the inundated area within the street segment using the zonal statistics tool of ArcGIS Pro.

The framework for PI LSTM surrogate model is shown in Figure 1
<img width="975" height="640" alt="image" src="https://github.com/user-attachments/assets/cdcc3e7a-dad5-48fe-ba41-39216af444b5" />
