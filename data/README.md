# Data Structures / Test Configuration

# Rock Tests
* If the rocks are above the ground, do not set them too high above the level of the sand. The optical flow camera throws errors with sudden changes in its view.
* Make sure an extremely pointy side is not faced up to prevent damage.
* Only travel over one rock at a time. Do not put rocks next to each other. Wheel should only contact one rock at a time.
* Randomize the placement along the pressure pad and the surface facing up.
* Make sure rocks are on the path of the pressure pad.

# Slip induce test
* Measure the depth of the extrusion into the sand relative to the cart. For instance put the the extrusion X cms from a certain point on the cart. This is how we chose to do it because the level of the sand is always changing and this gave us a sense of repeatability.
* Make sure the edges of sand in the trough are leveled as well. It is necessary to keep the level of the sand that the extrusion passes through the same, if not then the level of slip can vary for each test. For these tests only it is important to make the entire test bed level rather than only the path the wheel follows.
* It is necessary to decompact (turn over) the sand on every test, but for these test take special care to loosen the sand where the extrusions pass through. Varying levels of compactness can give you inconsistent slip values between tests. 
* One centimeter change in depth can increase slip rates by a lot.

# Pattern Tests
* To do patterns dug into the ground, we drew deep cuts into the sand using 8020 extrusions. To make patterns on the surface(rising off the surface of the sand) of the sand, we used excess sand and a shovel to pour, and draw on the patterns with the sand.
* Specifics of pattern testing will evolve as project continues.

# Pebbles
* Place pebbles lightly, especially in dust like, fine grain sands otherwise they will get lost beneath the surface.
* Be sure to get each pebble out of the sand.
* Use the sifter to find the pebbles. It may not work as well with coarser grained sands.
* Make sure the pebbles are in the path of the pressure pad.
* You will be told how to pattern them, dense, sparse, clusters, gradient etc.

# Rock Detection
* Bury the rocks the specific depth beneath the surface.
* dig a hole, grab the dirt leveling device and rest it on the trough above the hole, make sure all four wheels are in contact with the trough Get a calipers, and using the back edge of the calipers push the back arm out from the end and rest the thick part of the caliper on top edge of the lower portion of the U channel. So the skinny arm from the caliper can measure the distance of the  rock in the hole. The thick portion of the caliper should be resting on the u channel. When trying to bury the rock X cm beneath the surface, we like to bury it X +0.7 cm beneath the surface. This accounts for the thickness in the u channel and the fact that the u channel scrapes away the top layer of sand when flattening dirt. Ray has pictures of this. 
* Before covering the rocks with sand be sure to mark where the rock is (one dimensional) with the magnets. Flatten the dirt, then put the screws on each of the magnets so the side cameras can see where the rock is based on the screws and magnets
* Be sure the wheel completely rolls over the last rock and the test does not end while the wheel is resting on the rock

# Hydration 
* Take vial samples from the area that the hydration sensor passes through.
* Measure the  mass of the empty vials each time
* Bake for 15 minutes at 175 Celsius. And use convection bake
* Be sure not to spill them as you take them out. Ask Ray to make the wooden to hold the vials in place when they are baking


 ## Science & Engineering Target Supporting Papers
* [Martian Terrain Types](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/papers/Martian%20terrain%20types%20%26%20science%20interest%201-s2.0-S0022489816300891-main.pdf)
* [Conversion from Effective Size Mesh to mm](https://en.wikipedia.org/wiki/Mesh_(scale))
* [Mechanical soil measurement using sieve analysis](http://www.ce.memphis.edu/1101/notes/filtration/sieve_analysis.pdf)
* [Ground truth data](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/papers/CalTest%20JPL%20150115%20LAB%20TEST%20RESULTS%20prnt.pdf)

## [Mojave-1 Mars Simulant](https://github.com/JPLMLIA/Barefoot_Rover/tree/master//docs/papers/Simulant%20MMS%20Paper.pdf)
* Doesn't absorb as much water as competitors
* Have at least four grades: dust, 0.2mm, intermediate, coarse

| Simulant       | ~D60 (mm) |
|----------------|-----------|
| mmdust         | 0.025     |
| mm.2mm         | 0.04      |
| mmintr         | 0.22      |
| mmcrse         | >1 ?      |

## Red Garnet
* Extremely coarse but mixed with extremely fine dust and occasional organic components

| Simulant       | ~D60 (mm) |
|----------------|-----------|
| regdar         | > 1.5 ?   |

## M2020 Mobility simulants ([link1](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/Wheel%20Media%201%20Pamphlet%20v5%201%20p1.pptx),[link2](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/Wheel%20Media%201%20Pamphlet%20v5%20p2.pptx))
| Simulant       | Long Name | D60 (mm) | Shape   | Density (g/cc) | Friction Angle | Notes                    |Sourcing | 
|----------------|-----------|----------|---------|----------------|----------------|--------------------------|---------|
| grc-01         | GRC-1          | 0.39     | angular | 1.6            | 36             |wide dist size, well-known|Black Lab Corps (Chardon OH) |
| bst110         | Best Sands 110 | 0.08     | angular | 1.4            | 32             |finest, densifies as used |Black Lab Corps (Chardon OH) |
| mins30         | Minus 30       | 0.19     | angular | 1.4            | 33             |Mars Yard, cheap          |Soils Direct (Sunland CA) |
| wed730         | Wedron 730     | 0.14     | rounded | 1.5            | 33             |Only rounded option       |Black Lab Corps (Chardon, OH) |

## Simulant's Published Terramechanics Properties
| Simulant | Density (g/cc) | Cohesion _c_ (kPa) | Friction Angle (deg) | Shear Deformation _K_ (cm) | Source |Notes|
|----------|----------------|--------------------|----------------------|----------------------------|--------|-----|
|grc-01    |1.64            |0.28 (psi not kPa?)          |35.0                  |2.32                    |Oravec's Thesis |grousers on shear ring|
|grc-01    |1.75            |0.41 (psi not kPa?)         |33.5                  |2.00                    |Oravec's Thesis |grousers on shear ring|
|mms 'sand'|1.38            |0.81 (psi?)          |38.0                  |n/a                     |Peters          | |
|mms 'sand'|1.34            |1.96            |39.0                  |n/a                     |Peters          | |
|wed730    |1.52             |6.2            |30                      |0.05                    |S. Moreland            | |          
|bst110    |1.33             |6.55            |32                   |0.076                     |S. Moreland             | |
|mins30    |1.4             |5.17             |33                   |0.127                    |S. Moreland           | |

## Wheel Properties
* Track Length (2 rotations): 4m 
* Width: 0.45135 m
* Grouser width: 0.445 m
* Diameter (no grousers): 0.515 m
* Diameter (w/ grousers): 0.537 m
* Circumference (no grousers): 1.62 m
* Circumference (w/ grousers): 1.69 m

[[images/Wheel_Dimensions_MSL_M2020_Tactile.png|Comparative Wheel Dimensions]]
[[images/Tactile Wheel.gif|Wheel in Motion]]
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/WheelElectronics.PNG)

## CROSSBOW
Cart-based Robotic One-wheeled Surrogate System for Barefoot OSSE Work

### Requirements:
- Up-down sensorwheel motion freedom
- Driven by sensor wheel actuator (borrow actuators, constant velocity, straight track motion)
- Cart wheels are to rest on smooth floor beyond regolith tray or upon smooth rails in regolith
- Tray-based regolith system with linear, smooth tracks for cart wheel travel
- Weight-loading on sensor wheel realistic to rover config
- (4) travel velocities ranging from very slow (for a rover) to M2020 top speed
- Data alignment of all sensors with sampling cadence up to 10 Hz (variable)
   - Pressure grid
   - Actuator current
   - Force/torque sensor (borrowed)
   - EIS multi-frequency sweep
   - Sinkage gauge (linear track)
   - Visual camera system recording surface in contact with wheel (forward-focus)
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/RigOperation.gif)
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/EBoxAnnotated.PNG)

# List of Test Rig Components
##  Z Stage
* Allows vertical axis change.
* Allows the wheel move up and down freely without the entire test rig moving vertically.
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/axialchange.jpg?s=100)

## Cameras(3)
* All cameras point to different sides of cart
* Front, Back, Side
* Ensure cameras point as the wheel and dirt prior to running tests
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/camer1.jpg)
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/camera2.jpg)
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/camera3.jpg)

## PX4Flow Smart Camera
We have eliminated the PX4Flow Camera.

## AC to DC converter
* Converts and regulates power from the socket to an appropriate voltage
* Converts power from AC to DC
* Motor runs at 48V
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/converter.jpeg)
## DC to DC converter(2)
* Takes the 48V and steps it down to 12V
* There are two of these
* Connected to USB hubs
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/IMG_6170.JPG)
## Moisture Sensor
* Located on the Outside of the Wheel next to pressure pads
* measures moisture in the dirt
* Not in contact with ground between 65 and 155 degrees
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/moisturesensor.jpg)
## DC Motor
* Midwest Motion D33-655H-48V GP81-195 EI-512 (discontinued model)
* 5.7A rated current
* 9.7 rpm max speed
* Gear Ratio 195.26:1
* Gear Efficiency 70%
* Torque Constant 0.213 Nm/A
* Attaches to torque sensor
* Adjust speed by changing values of potentiometer
* Fast: 301 rpm for motor (100%), 1.54 rpm for wheel
* Slow: 113 rpm for motor (0%), 0.58 rpm for wheel
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/motor.jpg)
## Spider Coupling
* Power and torque transfer between two shafts
* See motor image
## Motor Controller
* ESCON 50/5
* Outputs current/speed data to Arduino
* Controls motor speed via integrated potentiometers (next to microUSB)
* Reads control switch state to determine direction and start/stop
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/power.jpg)
## Arduino(red)
* Reads Z-stage, ESCON outputs, PX4Flow, cart IMU (ADXL345)
* See image above
## National Instruments Multifunction I/O device
* Black box measures strain gauge resistance in F/T sensor
* White box processes data accordingly for computer
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/topcontroller.jpg)
## Emergency Stop Button
* Overrides the control button by cutting power to the motor
* Causes  cart to stop until button is twisted and released
*After button is released, control returns to grey controller
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/IMG_6168.JPG)
## Controller
* Controls the movement of the cart
*Toggle switch determines the direction of motion
*Hold down the side button to make cart move
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/IMG_6169.JPG)

## Wheel Calibration

### Dark Current Calibration Test

This test is used to see what the pressure pads read when there is no pressure being applied to them. To do this, you must lay the wheel on its side and take the grousers off. Rather than taking the grousers completely off, it is easier to remove one side of the wheel and pull the grousers back so they are no longer touching the pressure pad. To remove the side of a wheel, simply unscrew the sides and pull them off. You may need a screw driver to help you pry everything off. Run LL_Test as you normally would for a run. You do not need to run anything through anaconda, just have LL_Test write to a data file that is saved somewhere on your laptop. With the wheel on its side and the grousers off, take data for an hour. Afterwards, reassemble the wheel.

### New Pressure Pad Calibration

On a flat plank, the wheel is rolled overtop in both grouser-removed and grouser-equipped configuration. Use a long(ish) USB cable to connect the cart to the wheel , and have one person pull the cart at the same time as another person rolling the wheel (see figure below). 
![](https://github.com/JPLMLIA/Barefoot_Rover/tree/master/docs/pics/NewWheel_Calibration_Setup.jpg)

In order to avoid data anomalies caused by the edges of the plank in contact with the pad, roll the wheel down the plank offset to one edge of the pad, and then roll the wheel back offset to the opposite edge of the pad (emphasis on edge of the pad, not the wheel, though one edge of the pad is against the wheel edge). Ensure that before rolling back, the starting angle is reset to the starting angle of the forward roll. In this way, two runs are done for each angle (0, 90, 180, 270, labelled on the inside of the wheel). Incremented weights are 10, 20, and 30 lbs. This produces a calibration curve to fit the raw data to real load responses. 
The above is completed fully with the grousers removed and with the grousers attached, forming two sets of calibration data.

### Grouser On Calibration

This test is used to see how a constant pressure applied to grousers will be read by the pressure sensor. Use the white calibration tool for this task. You will have to run LL_Test from your laptop to get pressure pad data. With the wheel on its side, attach the white tool to the grouser. The tool should not be touching the pressure pad or any other grousers. Roll the wheel over so that the tool is on the ground supporting the weight of the wheel. Do your best to hold the wheel in place directly above the pressure pad without applying any external pressure. The best way to do this is to make sure the slot the grouser fits into is perpendicular to the floor. Run LL_Test for 10 seconds, you should be able to see increased pressure around that grouser alone. Do this for each grouser on the wheel. Mark where you start and which direction you continue in with tape. This will allow you to number the grousers (1-48). 

Tips: This is easy to do sitting down. Place the wheel far enough in front of you that you can place the tool and then roll it towards you without it hitting anything. Also, never hit stop scan on LL_Test. If you set the scan interval to 50 ms, you can set the frame count on the data capture to 1000 to get exactly 10 seconds of data. I usually set it to 1010 to have a little extra data just in case.

### Grouser Off Calibration

This test is used to apply a constant pressure on the pad between the grouser and see what it reads. This works the same as the On Grouser Calibration, but you need to use the red tool instead of the white tool. With the wheel on its side, place the tool in between two grousers. The tool should not be touching either grouser and the lip on either side of the wheel should not inhibit the tool from being placed directly on the pressure pad. Place a piece of tape on top of the wheel that lines up with the section of pressure pad you are using. Roll the wheel over so that the tool is on the ground supporting the weight of the wheel. You will probably need to hold the tool in place for most of this process. Do your best to hold the wheel in place directly above the pressure pad without applying any external pressure. Keep the piece of tape oriented perpendicular to the floor to make sure that the pressure pad is also oriented correctly. Run LL_Test for 10 seconds, you should be able to see increased pressure around that area alone. Do this for each section on the wheel. Mark where you start and which direction you continue in with tape. This will allow you to number the sections (1-48). 

## CROSSBOW Calibration

### String Potentiometer Calibration

To find the scaling factor: The potentiometer scaling factor is linear. All you need to do is find the voltage output relative to the distance for two different points. Calculate them as a slope and that will give you a scaling factor. Just to be sure, do a few more and average them. Use LJ control panel to get a voltage output. You will need to change the channel setting for higher voltages. Make sure that there is a tape measure attached to the trough before doing any calibration. Put the rig on the trough and attach the string pot as you would normally for a regular run. Make sure to account for the offset between where the tape measure reads zero and where the rig actually starts at zero. So, when the rig is at zero, the tape measure shows it is at 0.5", move the rig out to 85.5" so the string pot will read 85". This can be different every time.

For temperature calibration, use LJ Log to take long periods of data. You will need to download this software off of Lab Jack's website. Make sure that there is a tape measure attached to the trough before doing any calibration. Put the rig on the trough and attach the string pot as you would normally for a regular run. You do not need to attach the wheel. Move the rig out to whatever distance is specified by your mentor (generally 85"). Make sure to account for the offset between where the tape measure reads zero and where the rig actually starts at zero. So, when the rig is at zero, the tape measure shows it is at 0.5", move the rig out to 85.5" so the string pot will read 85". This can be different every time. Then start LJ Log, you should see readings for both the potentiometer and the temperature sensor. Make sure that the potentiometer's negative channel input is set to 32, that way the higher voltages can still be read. Leave the rig in that position for at least a few hours. You can use a heat gun to artificially heat up the string potentiometer as well.  

# CROSSBOW data format summary
Data from CROSSBOW is exported in the following separate streams, due to timing differences between the various platforms:

Directory with concatenated video frames from the monitoring webcams (each frame has UTC timing overlay)

 * Each frame named with the UTC timestamp so that they can be ordered/sorted by filename and referenced for time

ATI
 * __Nx7__ array: [UTC,Fx,Fy,Fz,Tx,Ty,Tz]
 * Uses the commpany-provided transforms to map to the force-torque cell reference frames
 * Also outputting an additional numpy array tagged with '_x' to account for additional transform aligning the cell to the cart reference frame a bit better

Cart Arduino
 * output from Arduino on cart, running: optical flow sensor (PX4Flow), Z-stage, cart IMU, available motor controller feedback 
 * __Nx10__ array: [UTC,Arduino timing,z-stage value (0.01 mm increment), ESCON current, ESCON speed, IMU x, IMU y, IMU z, PX4flow x (forward/back) displacement, PX4flow y (left/right) displacement]
    * ESCON motor controller outputs are scaled arbitrarily to values set on the board, which need to be known for any absolute meaning:
        * ESCON current and ESCON speed should be values in the 0-1023 range, output directly from the Arduino, reading an analog input in the 0-5V range
        * Current corresponds to 0-4V output for 0-4A motor current draw
        * Speed corresponds to 0-1.35V output for 0-1350rpm motor speed
        * From [motor datasheet](http://www.midwestmotion.com/products/brushed/48VOLT/10-24%20RPM/1300-1850%20IN-LBS/MMP%20D33-655E-48V%20GP81-195.pdf), motor torque constant is 14.3 oz-in/A, which can be used to approximate the motor torque under ideal conditions (which we do not have)
    * IMU xyz are accelerations, typically mapped (like on wheel) to actual rotation relative to gravity
    * PX4flow values are scaled relative to pixel count, not yet calibrated to real-world displacement
 * Arduino just spits out a tab-delimited string per measurement

Wheel Arduino
 * Unmodified from code used on Tactile Wheel
 * __Nx8__ array: [UTC,IMU x, IMU y, IMU z, r, theta, phi, rho]
    * Final rho value is typically the only one we've used for Tactile Wheel, denoting the rotation of the wheel itself. Any calibration between pressure sensor contact point and rotational value was calculated with respect to rho

Pressure Sensor
 * Proprietary .dat output files
 * Both Barefoot and Tactile Wheel have processing scripts that convert these text tables to more reasonable numpy format

Webcam Fiducial Odometry
 * Returns two data files \*_ao.npy and \*_ao_x.npy to record the fiducial and cart odometry
   * __Nx31__ array (\*_ao.npy): [UTC, x0,y0...x29,y29]
      * tracks the location of the possible fiducials (numbered 0-29) in image space
      * this is the raw fiducial data from the images, before processed to calculate cart displacement in real world coordinate frame
   * __Nx6__ array (\*_ao_x.npy): [UTC, dt, dx (pixel), dy (pixel), total x (mm), total y (mm)]  
      * tracks the relative change in fiducial displacement to provide overall cart motion. 
      * total y is the amount of displacement along the trough
 * Tracks relative motion of the Aruco fiducial tracks lined up on the side of the trough as the cart moves over them

Stringpot 
 * __Nx3__ array (\*_stringpot.npy): [UTC, voltage, total y (mm)]
 * Reports output from the Labjack module connected to the string potentiometer

Slip Calculations
 * __Nx6__ array: [UTC, 1 sec window, 2 sec window...5 sec window]
    * slip amounts calculated per time window previous to the UTC time
    * this convention was taken from the Scarecrow data logs. Choice of the time window for slip identification seems pretty important, and I'm not sure if there's some standard time window that is traditionally used for terramechanics analysis
 * Post-processed data files combines the odometry and wheel rotation data arrays to calculate slip for various time windows.
 * \*_slip.npy file reports slip using fiducial odometry, \*_slip_sp.npy file reports slip using the stringpot

Files/directories are named by the iso-format of the time cart initialization: [YYYY]-[MM]-[DD]T[HH]-[mm]-[ss][suffix]

* e.g. 2018-02-26T15-04-18_ati.npy
* files are numpy arrays

