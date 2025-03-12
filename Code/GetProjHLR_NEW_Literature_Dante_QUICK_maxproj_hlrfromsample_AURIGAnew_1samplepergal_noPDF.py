import numpy as np
import pynbody
import sys
import os
import pandas as pd
import warnings
import math
from scipy.interpolate import interp1d
from scipy import stats
import scipy

mapbins = 48

def new_bw(self):
        #Compute Scott's factor.
        return 0.5*self.neff**( -1./(self.d+4))

warnings.filterwarnings("ignore")

def fibonacci_sphere(samples=100):

	xlist = np.zeros(samples)
	ylist = np.zeros(samples)
	zlist = np.zeros(samples)
	phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

	for i in range(samples):
		y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
		radius = math.sqrt(1 - y * y)  # radius at y

		theta = phi * i  # golden angle increment

		x = math.cos(theta) * radius
		z = math.sin(theta) * radius

		xlist[i] = x
		ylist[i] = y
		zlist[i] = z
	return xlist, ylist, zlist

def half_light_r_count_XYZ(X,Y,Z, cylindrical=False):


    half_l = int(len(X) * 0.5)

    if cylindrical:
        R = np.sqrt(X**2+Y**2)
    else:
        R = np.sqrt(X**2+Y**2+Z**2)
        
    max_high_r = np.max(R)
    
    test_r = 0.5 * max_high_r
    
    testrf = R<test_r
    
    min_low_r = 0.0
    
    test_l = int(len(X[testrf]))
    
    it = 0
    while ((np.abs(test_l - half_l) / half_l) > 0.01):
        it = it + 1
        if (it > 100):
            break

        if (test_l > half_l):
            test_r = 0.5 * (min_low_r + test_r)
        else:
            test_r = (test_r + max_high_r) * 0.5
        testrf = R<test_r
        test_l = int(len(X[testrf]))

        if (test_l > half_l):
            max_high_r = test_r
        else:
            min_low_r = test_r

    return test_r
	
#################################################################################################
# DEFINE WORKING PATH, READ DATA AND MASK IT
#################################################################################################

main_file_folder = '/scratch/jsarrato/Wolf_for_FIRE/work/'
#main_file_folder = '/home/jorge/Physics/PhD/Paper3/Work/'

df = pd.read_csv(main_file_folder + 'file_labels_NIHAOandAURIGA.csv').drop_duplicates(subset='Galaxy').reset_index(drop=True)


# Step 2: Custom function to convert numbers
def convert_number(val):
    if isinstance(val, str):
        # Case where commas are used as decimal points
        if 'E' in val and ',' in val:
            val = val.replace(',', '.')  # Replace comma with dot
        try:
            return float(val)  # Attempt to convert to float
        except ValueError:
            return None  # Handle any conversion errors (e.g. non-numeric values)
    return val  # If it's not a string, return the value as-is


df['Mh'] = df['Mh'].apply(convert_number)
df['M_vir'] = df['M_vir'].apply(convert_number)

df = df.fillna(-9999)

# Step 1: Group by all columns and count occurrences
count_df = df.groupby(list(df.columns)).size().reset_index(name='n_proj')

# Step 2: Drop duplicates to keep the first occurrence and retain original order
ordered_df = df.drop_duplicates(keep='first')

# Step 3: Merge the counts back into the ordered dataframe
gal_data = pd.merge(ordered_df, count_df, on=list(df.columns), how='left')



"""gal_data = gal_data[gal_data['Galaxy'].str.endswith('h1')]
gal_data = gal_data.sort_values('m_r3').reset_index(drop = True)"""

paths = gal_data['path']

#################################################################################################
# DEFINE VARIBALES
#################################################################################################

n_processes = int(sys.argv[1])
n_process = int(sys.argv[2])

#n_processes = 1
#n_process = 1

N_sphere_points = 32


x,y,z = fibonacci_sphere(N_sphere_points)

n_project = N_sphere_points

filename = 'proj_data_NIHAOandAURIGA_noPDF' + str(n_process) + '.csv'

n_gals = len(paths)


startindex = int((n_process-1)*n_gals/n_processes)
endindex = int(n_process*n_gals/n_processes)
if n_process == n_processes:
   endindex = n_gals+1

count = n_project * startindex
current_path = ''
current_halo = ''

def fun(gkde):
     return gkde.n**(-1./(gkde.d+4))/3


r_array = np.arange(0.6,2.6,0.2)
r_array = np.append(r_array,np.array([1.7])) # Amorisco & Evans
r_array = np.append(r_array,np.array([10000])) # e.g Errani
r_array = np.append(r_array,np.array([4/3])) # Wolf
r_array = np.append(r_array,np.array([1.04])) # Campbell
rheader_list = list(['r'+str(ii+1) for ii in range(np.size(r_array))])
mrheader_list_circ = list(['m_r_circ'+str(ii+1) for ii in range(np.size(r_array))])
dispsheader_list_circ = list(['disp_r_circ'+str(ii+1) for ii in range(np.size(r_array))])

headers = ['path','Galaxy','i_index','count','hlr']
                    
n_projs_computed = 0
if os.path.exists(main_file_folder + filename):
  existing_csv = pd.read_csv(main_file_folder + filename)
  startindex = existing_csv['i_index']
  startindex = startindex[np.size(startindex)-1]+1
  count = existing_csv['count']
  count = count[np.size(count)-1]
  n_projs_computed = len(existing_csv[existing_csv['i_index']==startindex-1])
  
  N_projs_necessary = N_sphere_points

  if n_projs_computed < N_projs_necessary:
    startindex-=1
  else:
    n_projs_computed = 0

# For each copy of a galaxy
for i in range(startindex, endindex):
  x,y,z = fibonacci_sphere(N_sphere_points)

  n_project = N_sphere_points

  new_gal_data = pd.DataFrame(columns = headers)

  galname = gal_data['Galaxy'][i]
  
  if galname == 'g1.54e13h82':
    continue


  galpath = paths[i]

  print(galname)

  galname_original = galname

  halonumstr = galname[galname.rfind('h')+1:]
  halonum = int(halonumstr)
  galname = galname[:galname.rfind('h')]
  

  # if not already in memory, charge the simulation and center the galaxy
  if galpath != current_path:

    s = pynbody.load(galpath)

    s.physical_units()
    h = s.halos()
    
    current_path = galpath
    current_halo = ''
    
    print('Charged '+galpath)

  if current_halo != halonumstr:
    print('Centering halo '+halonumstr)
    try:
      h1 = h[halonum]
      
      pynbody.analysis.halo.center(h1)

      try:
        h1 = h1[h1['r']<h1.properties['Rvir']]
      except:
        h1 = h1[h1['r']<h1.properties['Rhalo']]

      try:
        h1.s = h1.s[h1.s['aform']>0.]
      except:
        try:
          h1.s = h1.s[h1.s['age']>0.]
        except:
          print('Could not separate wind particles')


      try:
        pynbody.analysis.angmom.faceon(h1, use_stars = True)
      except:
        pynbody.analysis.angmom.faceon(h1, use_stars = False)


      current_halo = halonumstr

      centered = True
      print('Centering done')
    except:
       centered = False
       print('Centering failed')
  else:
    print('Halo '+halonumstr+'already centered')

  n_stars = min(max(int(len(h1.s['x'])/20),200),20000)
  if n_stars > len(h1.s['x']):
    n_stars = len(h1.s['x'])
  #n_stars = len(h1.s['x'])
  if n_stars<100:
     centered = False
     
  Xmeanvel = np.zeros((N_sphere_points,mapbins,mapbins))
  Xcount = np.zeros((N_sphere_points,mapbins,mapbins))
  XDMmass = np.zeros((N_sphere_points,mapbins,mapbins))
  Xstd = np.zeros((N_sphere_points,mapbins,mapbins))

  """
  try:
    indeces2 = np.random.choice(np.arange(0,len(h1.s['x'])), size = n_stars, replace = False)
    #indeces = np.argsort(h1.s['r'])[:n_stars]
    #np.random.shuffle(indeces2)
    indeces = indeces2
    #indeces = np.arange(0,n_stars)
  except:
    print('less than 200 stars?')
    indeces = np.arange(0,len(h1.s['x']))
  """

  try:
    indeces2 = np.random.choice(np.arange(0,len(h1.s['x'])), size = n_stars, replace = False)
    #indeces = np.argsort(h1.s['r'])[:n_stars]
    #np.random.shuffle(indeces2)
    #indeces = indeces2
    #indeces = np.arange(0,n_stars)
    #indeces = np.arange(0,len(h1.s['x']))
  except:
    print('less than 200 stars?')
    indeces = np.arange(0,len(h1.s['x']))



  for project in range(n_projs_computed,n_project):

      print(n_stars, len(h1.s['x']))
      XYZ_0 = np.vstack((h1.s['x'],h1.s['y'],h1.s['z']))
      VXYZ_0 = np.vstack((h1.s['vx'],h1.s['vy'],h1.s['vz']))

      XYZ_0_dm = np.vstack((h1.d['x'],h1.d['y'],h1.d['z']))
      #VXYZ_0_dm = np.vstack((h1.d['vx'],h1.d['vy'],h1.d['vz']))
      M_dm = h1.d['mass']


      new_gal_data = pd.DataFrame(columns = headers)
      print(str(i+1)+'/'+str(endindex)+' , '+str(project+1)+'/'+str(n_project),count)
      if centered:

          count+=1

          nv = np.array([x[project], y[project], z[project]])  # Replace a, b, c with your normal vector components
          nv /= np.linalg.norm(nv)  # Normalize the normal vector

          rotmatrix = np.array([[nv[1]/np.sqrt(nv[0]**2 + nv[1]**2), -nv[0]/np.sqrt(nv[0]**2 + nv[1]**2) ,0],
                                [nv[0]*nv[2]/np.sqrt(nv[0]**2 + nv[1]**2), nv[1]*nv[2]/np.sqrt(nv[0]**2 + nv[1]**2), -np.sqrt(nv[0]**2 + nv[1]**2)],
                                [nv[0], nv[1], nv[2]]])



          XYZ = np.matmul(rotmatrix, XYZ_0)
          X = XYZ[0,:]
          Y = XYZ[1,:]

          RXY = np.sqrt(X**2+Y**2)

          VXYZ = np.matmul(rotmatrix, VXYZ_0)
          VZ = VXYZ[2,:]


          XYZ_dm = np.matmul(rotmatrix, XYZ_0_dm)
          X_dm = XYZ_dm[0,:]
          Y_dm = XYZ_dm[1,:]

          #try:
          hlr_proj_cumsum = half_light_r_count_XYZ(X[indeces2],Y[indeces2],Y[indeces2],cylindrical=True)
          r_arr1 = hlr_proj_cumsum*r_array

          indeces = (np.sqrt(X**2 + Y**2) < 0.5*hlr_proj_cumsum)
          if len(indeces[indeces]) < n_stars:
              n_stars = len(indeces[indeces])
          indeces_new = np.random.choice(np.arange(0,len(indeces[indeces])), size = n_stars, replace = False)

          X = X[indeces][indeces_new]
          Y = Y[indeces][indeces_new]
          VZ = VZ[indeces][indeces_new]
          
          xmin = -1
          xmax = 1
          ymin = -1
          ymax = 1

          m1 = X/hlr_proj_cumsum
          m2 = Y/hlr_proj_cumsum
          m3 = VZ

          m1_dm = X_dm/hlr_proj_cumsum
          m2_dm = Y_dm/hlr_proj_cumsum

          Xmeanvel[project,:,:] = scipy.stats.binned_statistic_2d(m1, m2, m3, statistic='mean', bins=mapbins, range=[[xmin,xmax],[ymin,ymax]]).statistic.reshape((1,mapbins,mapbins))
          Xstd[project,:,:] = scipy.stats.binned_statistic_2d(m1, m2, m3, statistic='std', bins=mapbins, range=[[xmin,xmax],[ymin,ymax]]).statistic.reshape((1,mapbins,mapbins))
          Xcount[project,:,:] = scipy.stats.binned_statistic_2d(m1, m2, m3, statistic='count', bins=mapbins, range=[[xmin,xmax],[ymin,ymax]]).statistic.reshape((1,mapbins,mapbins))

          XDMmass[project,:,:] = scipy.stats.binned_statistic_2d(m1_dm, m2_dm, M_dm, statistic='sum', bins=mapbins, range=[[xmin,xmax],[ymin,ymax]]).statistic.reshape((1,mapbins,mapbins))



          n_stars = len(X)
          print(n_stars)

      else:
          r_arr1 = np.ones_like(r_array)*-9999

          hfile = open(main_file_folder + '/Logs/'+galname+'_CenterError.dat','w')
          hfile.write(galname+'  '+str(halonum)+'  '+galpath+'  ')
          hfile.close()

      gal_result = [galpath, galname_original ,i, count, hlr_proj_cumsum]
      gal_dict = {headers[i]: [gal_result[i]] for i in range(len(headers))}
      gal_line = pd.DataFrame(gal_dict)
      new_gal_data = pd.concat([new_gal_data, gal_line])

      if os.path.exists(main_file_folder + filename):
        new_gal_data.to_csv(main_file_folder + filename, index = False, mode = 'a', header = False)
      else:
        new_gal_data.to_csv(main_file_folder + filename, index = False, mode = 'w')
          
  n_projs_computed = 0


  np.save(main_file_folder + 'Maps_Dispersion_NIHAO_noPDF_DMmap/file_Xstd'+str(i), Xstd)

  np.save(main_file_folder + 'Maps_Dispersion_NIHAO_noPDF_DMmap/file_Xvel'+str(i), Xmeanvel)

  np.save(main_file_folder + 'Maps_Dispersion_NIHAO_noPDF_DMmap/file_Xcount'+str(i), Xcount)

  np.save(main_file_folder + 'Maps_Dispersion_NIHAO_noPDF_DMmap/file_XDMmass'+str(i), XDMmass)
