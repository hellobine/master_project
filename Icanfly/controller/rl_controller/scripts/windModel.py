import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import torch
from scipy.interpolate import interp1d
from scipy.stats import vonmises
import math


class Wind:

    def __init__(self, num_envs, eval=False, dt=0.01, episodeLengthSeconds=15, speedMPS=2, directionVectorEarthFrame=np.array([1,0,0]), isSpeedVariable="constant", 
    speedSinusoidFrequencyHz=0.5, speedSinusoidAmplitude=1, isDirectionVariable="constant", directionSweepLimitDegrees=45, 
    directionSweepFrequencyHz=0.25, seed=None):
        
        self.windFlag = 0
        self.num_envs = num_envs
        self.variableSpeedTypes = ["constant", "sinusoidal", "realistic"]
        self.variableDirectionTypes = ["constant", "sweep", "realistic"]
        
        self.isSpeedVariable = isSpeedVariable.lower()
        self.episodeLengthSeconds = episodeLengthSeconds
        self.speedMPS = speedMPS # requested mean speed
        self.speed = speedMPS # sampled/current mean speed
        self.directionUnitVectorEarthFrame = directionVectorEarthFrame / np.linalg.norm(directionVectorEarthFrame)
        self.speedSinusoidFrequencyHz = speedSinusoidFrequencyHz
        self.speedSinusoidAmplitude = speedSinusoidAmplitude
        self.isDirectionVariable = isDirectionVariable.lower()
        self.directionSweepLimitDegrees = directionSweepLimitDegrees
        self.directionSweepFrequencyHz = directionSweepFrequencyHz

        self.density = 1.225 #kg/m^3
        self.kinematicViscosity = 17.89e-6 #N s/m^2 @Pa @T=15degreesCelsius
        self.timeSeconds = 0
        self.dt = dt

        self.rng = np.random.RandomState(seed)

        self.droneBaseQuaternion = torch.zeros((self.num_envs,4), device="cpu", dtype=gs.tc_float)

        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.e = 0
        self.f = 0 #coefficients for random wind initialization

        if not eval:
            self.resampleCoefficients()

        #self.directionUnitVectorEarthFrame = self.angleToUnitVectorZ(self.b)

        assert self.isSpeedVariable in self.variableSpeedTypes
        assert self.isDirectionVariable in self.variableDirectionTypes

    #def testprint(self):
    #    print("success")

    def resampleCoefficients(self):
        self.a = np.clip(self.rng.normal(self.speedMPS, 1), 0.5, 8.0)
        self.b = self.rng.uniform(0, 360)
        self.c = np.clip(self.rng.normal(1, 1), 0.5, 2.5)
        self.d = np.clip(self.rng.normal(0.7, 1), 0.25, 1.5)
        self.e = np.clip(self.rng.normal(40, 20), 15, 80)
        self.f = np.clip(self.rng.normal(0.4, 1), 0.125, 1.0)
        self.recalculateParams()

    def recalculateParams(self):
        self.speed = self.speedMPS#self.a
        self.directionUnitVectorEarthFrame = self.angleToUnitVectorZ(self.b)
        #self.speedSinusoidAmplitude = self.c
        #self.speedSinusoidFrequencyHz = self.d
        #self.directionSweepLimitDegrees = self.e
        #self.directionSweepFrequencyHz = self.f

        print("Wind Speed: " + str(self.speed))
        print("Angle: " + str(self.b))
        print("VariableSpeed: " + self.isSpeedVariable)
        print("VariableDirection: " + self.isDirectionVariable)
        #print(self.windFlag)

    def calculateWindForce(self, drone):
        if self.timeSeconds==0:
            if self.isSpeedVariable == "realistic":
                print("success")
                self.realisticSpeed, self.realisticDirection = self.realisticWind()

        dragCoefficient = 1.2 #for a flat plate orthogonal to flow
        areaMetres = torch.tensor([0.092*0.029, 0.092*0.029, 0.092*0.092], dtype=gs.tc_float) #m^2 [A_xx, A_yy, A_zz]
        velocityRelativeBody = self.calculateVelocityRelativeBody(drone)
        #print("vrel")
        #print(velocityRelativeBody)
        windForceBody = torch.tensor((-1/2)*(self.density)*(dragCoefficient)*areaMetres*(np.abs(velocityRelativeBody))*(velocityRelativeBody), dtype=gs.tc_float)
        #print("Fb")
        #print(windForceBody)
        #print("body")
        #print(windForceBody)
        windForceEarth = transform_by_quat(windForceBody, inv_quat(self.droneBaseQuaternion))
        #print(windForceEarth)    
        self.timeSeconds += self.dt
        #print("Fe")
        #print(windForceEarth)
        return windForceEarth

    def speedSinusoidMPS(self):
        return self.speed + (self.speedSinusoidAmplitude * np.sin(2*np.pi*self.speedSinusoidFrequencyHz*self.timeSeconds))

    def directionSinusoidRadians(self):
        return (self.directionSweepLimitDegrees*(np.pi / 180)) * np.sin(2*np.pi*self.directionSweepFrequencyHz*self.timeSeconds)

    def realisticWind(self, n_turbulence_components=5):
        """
        Generate realistic free-space wind using a combination of:
        1. Turbulent fluctuations using simplified spectrum
        2. Wind gusts using random walk
        3. Von Mises distribution for more realistic direction changes
        """
        # Base wind parameters
        mean_speed = max(0, self.rng.normal(8, 2))  # mean wind speed
        mean_speed = min(mean_speed, 10)  # cap mean speed at 10 m/s
        mean_direction = self.rng.uniform(0, 360)
        
        # Generate time array
        t = np.arange(0, self.episodeLengthSeconds, self.dt)
        
        # 1. Generate turbulent components using simplified spectrum
        def simplified_spectrum(f):
            """Simplified spectrum for free-space wind fluctuations."""
            return 1.0 / (1 + np.abs(f))
        
        # Generate frequencies for spectrum
        freqs = np.fft.fftfreq(len(t), self.dt)
        freqs[0] = 1e-6  # avoid division by zero
        
        # Generate turbulent components
        turbulence = np.zeros_like(t)
        for _ in range(n_turbulence_components):
            amplitude = np.sqrt(simplified_spectrum(np.abs(freqs)))
            phase = 2 * np.pi * self.rng.random(len(freqs))
            component = np.fft.ifft(amplitude * np.exp(1j * phase)).real
            turbulence += component
        
        # Scale turbulence to a fraction of the mean speed
        turbulence *= 0.2 * mean_speed
        
        # 2. Generate wind gusts using random walk
        gust_component = np.zeros_like(t)
        gust_strength = self.rng.normal(0, 0.5, len(t))
        for i in range(1, len(t)):
            gust_component[i] = gust_component[i-1] + gust_strength[i]
        gust_component -= np.mean(gust_component)
        gust_component *= 0.3 * mean_speed  # scale gusts relative to mean speed
        
        # Combine speed components
        wind_speed = mean_speed + turbulence + gust_component
        wind_speed = np.clip(wind_speed, 0, 10)  # ensure speeds are between 0 and 10 m/s
        
        # 3. Generate direction variations using von Mises distribution
        kappa = 8.0  # concentration parameter
        direction_variations = vonmises.rvs(kappa, loc=0, size=len(t))
        wind_direction = (mean_direction + np.degrees(direction_variations)) % 360
        
        # Convert directions to unit vectors
        windDirs = np.zeros((len(wind_direction), 3))
        for i in range(len(wind_direction)):
            windDirs[i] = self.angleToUnitVectorZ(wind_direction[i])
        
        return wind_speed, windDirs

            
    def calculateWindVelocityEarth(self):
        if self.isSpeedVariable == "constant":
            speed = self.speed
        if self.isSpeedVariable == "sinusoidal":
            speed = self.speedSinusoidMPS()
        if self.isSpeedVariable == "realistic":
            print(self.realisticSpeed[int(self.timeSeconds*(1/self.dt))])
            print(self.realisticDirection[int(self.timeSeconds*(1/self.dt))])
            return torch.tensor(self.realisticSpeed[int(self.timeSeconds*(1/self.dt))]*self.realisticDirection[int(self.timeSeconds*(1/self.dt))], dtype=gs.tc_float)

        if self.isDirectionVariable == "constant":
            direction = self.directionUnitVectorEarthFrame
        if self.isDirectionVariable == "sweep":
            direction = self.directionUnitVectorEarthFrame @ self.rotationMatrix(0,0,self.directionSinusoidRadians()) 
        
        return torch.tensor(speed * direction, dtype=gs.tc_float)

    def calculateVelocityRelativeBody(self, drone):
        self.droneBaseQuaternion[:] = drone.get_quat()
        #print(self.droneBaseQuaternion.shape)
        #print(self.calculateWindVelocityEarth().shape)
        windVelocityBody = transform_by_quat(self.calculateWindVelocityEarth().unsqueeze(0), self.droneBaseQuaternion)
        #print("windVb")
        #print(windVelocityBody)
        droneVelocityBody = drone.get_vel()
        #print("droneVb")
        #print(droneVelocityBody)
        velocityRelativeBody =  droneVelocityBody - windVelocityBody
        #print(velocityRelativeBody)
        return velocityRelativeBody

    def rotationMatrix(self, phi, theta, psi):
        """
        Returns the rotation matrix R for given Euler angles (phi, theta, psi).
        
        Parameters:
        phi : float  -> Rotation about the X-axis (in radians).
        theta : float -> Rotation about the Y-axis (in radians).
        psi : float   -> Rotation about the Z-axis (in radians).

        Returns:
        np.array (3x3) -> The rotation matrix R.
        """
        # Rotation matrix around X-axis (Roll)
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(phi), -np.sin(phi)],
                    [0, np.sin(phi),  np.cos(phi)]])
        
        # Rotation matrix around Y-axis (Pitch)
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
        
        # Rotation matrix around Z-axis (Yaw)
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi),  np.cos(psi), 0],
                    [0, 0, 1]])
        
        # Combined rotation matrix: R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx  # Matrix multiplication (order: X → Y → Z)
        
        return R
    
    import math

    def angleToUnitVectorZ(self, angle_deg):
        """
        Convert an angle about the z-axis into a unit vector pointing in the same direction.
        
        Parameters:
            angle_rad (float): The angle in radians, measured counterclockwise from the positive x-axis.
        
        Returns:
            tuple: A unit vector (x, y, z) representing the direction.
        """
        angle_rad = math.radians(angle_deg)
        x = math.cos(angle_rad)
        y = math.sin(angle_rad)
        z = 0  # The vector lies in the xy-plane
        return np.array([x, y, z])

