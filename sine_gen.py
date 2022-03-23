import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn import mixture

def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data

def plot_samples(S, n_samples, n_components): 
    
    for i in range(5):
        plt.plot(S[i])
    plt.title(f'{n_samples} generated samples with n_components = {n_components}')
    plt.show()

def learn_and_sample(X, n_components, n_samples):
  plt.figure()
  gmm = mixture.GaussianMixture(n_components=n_components).fit(X)
  S, labels = gmm.sample(n_samples)
  return S

X = np.array(sine_data_generation(10000, 30, 10))
X = X.reshape(1000, -1)
S = learn_and_sample(X, 15, 5)

for sample in X[:4]:
    plt.plot(sample[:100])
plt.title(f'real distribution')
plt.show()

for sample in S[:4]:
    plt.plot(sample[:100])
plt.title(f'generated distribution')
plt.show()