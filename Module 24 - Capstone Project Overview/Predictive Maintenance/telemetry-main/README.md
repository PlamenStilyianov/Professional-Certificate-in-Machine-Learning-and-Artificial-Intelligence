<center>
    <center>
        <img src = images/lstm.png width = 35%/>
    </center>
</center>

## Data Set Information:

The data belongs to a large cloud provider. It is obtained from their Data Center Backend Telemetry team. It is synthetic in nature and anonymized to obfuscate identifiable information.


#### Business Understanding

##### What is Predictive Maintenance?

Our world is full of equipment. For example:
- Aircraft consists of different equipments
- HVAC (Heating, Ventilation, Air Conditioning) equipment consists of various parts
- Similarly data centers run thousands of servers that need to be proactively and predictively maintained to avoid downtime 
  that can lead to business disruption, financial impact, and loss of productivity.

All these equipments or their parts meet failures and hence need maintenance. Thus maintenance is a big industry by itself. Maintenance is done either by replacing parts at a regular intervals even when those are working (**Preventive Maintenance**) or by replacing the parts only when there is a failure (**Reactive Maintenance**). **Predictive Maintenance** avoids the drawbacks of Preventive Maintenance (under utilization of a part's life) and Reactive Maintenance (unscheduled downtime). Based on the health of an equipment in the past, future point of failure can be predicted in Predictive Maintenance. Thus, replacement of parts can be scheduled just before the actual potential failure. Traditionally, predictive maintenance is being done using rule based techniques. With the advent of connected sensors (IoT), data from equipment is continuously collected and fed to Machine Learning based systems to predict its future health.

This is a classic case of unsupervised learning. Unsupervised learning can be compared to the way children learn about the world without the insights of adult supervision. No one teaches children to be surprised and curious about anything they’ve not seen before. Unsupervised learning means you’re only exposing a machine to input data. There is no corresponding output data to teach the system the answers it should be arriving at.

In our use case, we do not know about a specific state at which a machine can be characterized as faulty or a component requires to be replaced in order to keep the machine in the running state. Hence we need to follow unsupervised learning method to determine this. We'll be using Neural network paradigm of machine learning to make intelligent decisions with limited human assistance to determine the maintenance predictively. 

##### Why Neural network?

As we said before, we have no prior knowledge or experience of perceived failures. If there are events that are falling outside of our experiences, the only way to graps those events to make informed decisions in future is by way of building complex hierarchies of meaning to express information from the raw data. Neural networks are best suited to solve these types of problems.


#### Source:

A Large cloud provider

## Data Understanding
<pre>
Data Set Characteristics:  Multivariate
Area: IoT/Telemetry
Attribute Characteristics: Real
Missing Values? None
</pre>

#### Attribute Information:

There are 8 CSV files. 
- 5 files contain data center wide telemetry events
- 3 files contain actual rack/machine telemetry events

First 5 CSV files consisting of:

- **Telemetry Time Series Data** (telemetry.csv): It consists of hourly average of voltage, rotation, pressure, vibration collected from 100 machines for the year 2015.

- **Error** (errors.csv): These are errors encountered by the machines while in operating condition. Since, these errors don't shut down the machines, these are not considered as failures. The error date and times are rounded to the closest hour since the telemetry data is collected at an hourly rate.

- **Maintenance** (maint.csv): If a component of a machine is replaced, that is captured as a record in this table. Components are replaced under two situations:
    - During the regular scheduled visit, the technician replaced it (Proactive Maintenance)
    - A component breaks down and then the technician does an unscheduled maintenance to replace the component (Reactive  Maintenance). This is considered as a failure and corresponding data is captured under Failures.
    Maintenance data has both 2014 and 2015 records. This data is rounded to the closest hour since the telemetry data is collected at an hourly rate.

- **Failures** (failures.csv): Each record represents replacement of a component due to failure. This data is a subset of Maintenance data. This data is rounded to the closest hour since the telemetry data is collected at an hourly rate.

- **Metadata of Machines** (machines.csv): Model type & age of the Machines.

3 CSV file consist of - 

- **Metadata of attached hardware of individuals machines** (train.txt, test.txt, truth.txt - the names are slightly misleading)

## Team collaboration - directory structure

#### Instructions
<pre>
- Clone the GitHub repository
- Please run the notebooks in sequence
- At the end of the step 3, there will be 2 model files created.

├── data
│    ├── machines.csv   
│    ├── maint.csv  
|    ├── telemetry.csv
|    ├── errors.csv
|    ├── failures.csv
|    ├── train.txt
|    ├── test.txt
|    ├── truth.txt
├── models
|    ├── predictive_maint.hd5 
|    ├── predictive_maint.json
├── 1. Predictive-maintenance_exploratory_data_analysis.ipynb
│   2. Predictive_maintenance_machine_failure.ipynb
|   3. Predictive_maintenance_component_failure.ipynb
├── presentation
|   ├── Predictive_Maintenance_using_ML.pptx

</pre>

## Data Preparation and Visulalization
<pre>
Code Used: Python
Packages: Pandas, Numpy, Matplotlib
</pre>
## Data Cleansing, processing, and modeling
<pre>
Code Used: Python
Packages: Pandas, tensorflow, Keras (Sequential, Dense, Dropout, lambda, Flatten, LSTM, Activation)
</pre>
## Process Summary
**By performing different ML models, we aimed to get better at predictively optimizing following**
- operational efficiency
- inventory optimization
- just-in-time procurement
- warehouse cost
- predictable consistent uptime
- minimize SLA violations


<center>
    <img src = images/copyright.png width = 15%, align = "right"/>
</center>
