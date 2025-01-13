
import openseespy.opensees as ops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy 
import logging 


class OpenSeesModel: 

    def __init__(self, load_ts_fp, ndm=1, ndf=1):
        self.ndm = ndm 
        self.ndf = ndf 
        self.load_ts_fp = load_ts_fp

        print('*** Initializing Model ***')
        ops.wipe()
        ops.model(
            'basic',
            '-ndm',self.ndm,
            '-ndf',self.ndf
        )
        
    @staticmethod
    def construct_lollipop():
        print('*** Building Lollipop Structure w SDOF (Nodes/Links/Damping/Materials) ***')

        # Construct nodes
        ops.node(1,0.0,0.0);
        ops.node(2,0.0,1.0);

        # Assign mass
        ops.mass(2, 0.1)

        # Define Damping
        ops.rayleigh(2, 0.0, 0.0, 0.0)

        # Boundary conditions
        ops.fix(1,1);

        # Define Material(s)
        ops.uniaxialMaterial('Elastic', 1, 5)

        # Define element(s)
        ops.element(
            'twoNodeLink', 1, *[1,2], 
            '-mat', 1, 
            '-dir', 1
        )

    def generate_ts(self, interval=0.02, plot=True):
        print('*** Reading Ground Motion Data From Source ***')

        #copy the code from hw5 to read the PEER db file...
        if self.load_ts_fp == 'A100000.dat': 
            ops.timeSeries(
                'Path', 1, 
                '-dt', interval, 
                '-filePath', f'loadfiles/{self.load_ts_fp}', 
                '-factor', 386.
            ) 
        elif self.load_ts_fp.split('.')[-1] == 'AT2': 
            with open(f'loadfiles/{self.load_ts_fp}') as ts:
                ts_only = ts.readlines()[4:]
                ts_flattened = [
                    float(i) for j in ts_only for i in j.split(' ') if len(i) > 0
                ]
                ts.close()

            t = (np.arange(len(ts_flattened))*interval).tolist()
            ops.timeSeries(
                'Path', 1, 
                '-dt', interval, 
                '-values', *ts_flattened, 
                '-time', *t, 
                '-factor', 386.
            )
        else:
            raise Exception()
        
        if plot: 
            plt.figure(figsize=(10,5))
            plt.plot(
                t, 
                ts_flattened, 
            )
            plt.title('Ground Motion Acceleration (g)')
            plt.xlabel('Time (s)')
            plt.ylabel('Horizontal Accel. (g)')
            plt.savefig('outputs/ground_motion.png')
            plt.close()

        ops.pattern(
            'UniformExcitation', 1, 1, 
            '-accel', 1, 
            '-factor', 3
        )

        self.interval = interval
        self.n = max(t)
       

    def perform_analysis(self, tolerance=0.001, iterations=1000):

        print('*** Running Analysis ***')
        ops.wipeAnalysis()
        ops.constraints('Plain')
        ops.numberer('Plain')
        ops.system('ProfileSPD')
        ops.integrator('Newmark', 0.5, 0.25)
        ops.algorithm('Newton')
        ops.analysis('Transient')
        ops.test('NormUnbalance', tolerance, iterations)   

    def analyze_and_record(self, figure_name):
        self.perform_analysis()

        analysis_interval = self.interval*0.1
        ttl = 0
        numIncr = int(self.n/analysis_interval)
        data = np.zeros((numIncr+1,2))
        ops.setTime(0)
        for i in range(numIncr):
            ops.analyze(1,analysis_interval)
            data[i+1,1] = ops.nodeDisp(2,1)
            data[i+1,0] = ops.getTime()
            ttl += analysis_interval 
        print(f'Total Simulated Time (s): {ttl}')
        plt.figure(figsize=(12,5))
        plt.plot(data[:,0], data[:,1])
        plt.title(f'Displacement Simulation: {figure_name}')
        plt.xlabel('Time [sec]')
        plt.ylabel('Displacement [in]')
        plt.grid()
        plt.savefig(f'outputs/{figure_name}.png')
        plt.close()

    def simulate(self, figure_name):
        self.construct_lollipop()
        self.generate_ts()
        self.analyze_and_record(figure_name)

class WindLoadModel(OpenSeesModel):

    @staticmethod
    def force_conversion(pressure_coef, v_3s=119, length_scale=1/400, kips_conversion_factor=0.000224809):
        v_hr_bld = 0.7 * v_3s
        density = 1.225
        pressure = 0.5 * pressure_coef * density * (v_hr_bld ** 2)
        area = (1/length_scale)**2 * (1/254)**2 * 0.02**2  #dimensions of building faces
        return pressure * area * kips_conversion_factor
    
    def read_load_file(self, v_3s=119, length_scale=1/400, velocity_scale=1/5):
        assert (self.load_ts_fp.split('.')[-1] == 'mat')
        try:
            data = scipy.io.loadmat(f'loadfiles/{self.load_ts_fp}')
        except Exception as e:
            print(e)
            return []
        
        pressure_coefs = pd.DataFrame(data['Wind_pressure_coefficients'])
        pressure_coefs.columns = np.arange(1, pressure_coefs.shape[1]+1)

        ### as per the README, cols 3 and 4 map pressure tap points to directions of the building (rightward, leftward, leeward etc.)
        pressure_tap = pd.DataFrame(data['Location_of_measured_points'][2:,:].T, columns=['point','direction'])
        mapped_points=dict()
        for s in pressure_tap['direction'].unique():
            mapped_points[s] = pressure_tap.loc[pressure_tap['direction']==s]['point'].astype(int).to_list()
 
        #set to negative, counteracting forces from the back of the building?
        cross_pr = mapped_points[2] - mapped_points[4]
        along_pr =  mapped_points[1] - mapped_points[3]

        pressure_coefs['Along_wind_pr_coef'] = pressure_coefs[along_pr].sum(axis=1)
        pressure_coefs['Cross_wind_pr_coef'] = pressure_coefs[cross_pr].sum(axis=1)

        time_scale = length_scale/velocity_scale
        idx_increments = (1/time_scale)*0.001 
        pressure_coefs['Time (s)'] = pressure_coefs.index * idx_increments
        along_cross_ttls = pressure_coefs[['Time (s)', 'Along_wind_pr_coef', 'Cross_wind_pr_coef']]
        along_cross_ttls = along_cross_ttls.set_index('Time (s)')
        along_cross_ttls['Force_along_wind'] = along_cross_ttls['Along_wind_pr_coef']\
            .apply(
                lambda cp: self.force_conversion(cp, v_3s, length_scale)
        )
        along_cross_ttls['Force_cross_wind'] = along_cross_ttls['Cross_wind_pr_coef']\
            .apply(
                lambda cp: self.force_conversion(cp, v_3s, length_scale)
        )
        self.load_history = along_cross_ttls
        self.load_history.to_csv('outputs/load_history.csv')
        self.interval = idx_increments
        self.n = max(self.load_history.index)

    def generate_ts(self, face, id, plot=True):
        
        assert hasattr(self, 'load_history')
        if face in self.load_history.columns: 
            vals = self.load_history[face].tolist() 
            t = self.load_history.index.tolist()
            ops.timeSeries(
                'Path', id, 
                '-values', *vals, 
                '-time', *t, 
                '-factor', 386.
            )
            ops.pattern(
                'Plain', id, id, 
            )
            ops.load(2, 1)

        else: 
            raise Exception()
        
        if plot: 
            plt.figure(figsize=(12,5))
            plt.plot(
                t, 
                vals
            )
            plt.title(f'Wind Pressure History: {face}')
            plt.xlabel('Time (s)')
            plt.ylabel('Force [kn]')
            plt.savefig(f'outputs/pressure_{face}.png')
            plt.close()
 
    def simulate(self, figure_name):
        self.construct_lollipop()
        self.read_load_file()
        for id, face in enumerate(['Force_along_wind', 'Force_cross_wind']):
            self.generate_ts(
                id = id, face = face
            )
            self.analyze_and_record(figure_name+f'_{face}')
               
if __name__ == '__main__':
    earthquake_file = 'RSN942_NORTHR_ALH090.AT2'
    osm = OpenSeesModel(earthquake_file)
    osm.simulate(
        figure_name = 'earthquake'
    )

    wind_file = 'time_series_of_point_wind_pressure_0_1_4.mat'
    osm_wind = WindLoadModel(wind_file)
    osm_wind.simulate(
        figure_name = 'wind'
    )




        



        

