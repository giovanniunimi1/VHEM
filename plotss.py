import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
#PLOT WITHOUT IMG !

#cosa serve : plot_compact delle vbhmm, plot emission


#vhem_plot_clusters : per ogni gruppo di hmms, vbhmm plot compact di standardize

def hmm_plot(hmm, p,i):
    face_image = plt.imread("img_test.jpeg")
    face_height, face_width, _ = face_image.shape
    screen_resolution = [1920,1080]

    # Creazione della figura e dei subplot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    if p == 'g':
        fig.suptitle(f'Reduced model group {i}', fontsize=16)
    else :
        fig.suptitle('Base model ', fontsize=16)
    colors = ['red', 'green', 'blue', 'orange', 'purple'] 
    # Primo subplot: Immagine della faccia con ellissi
    axs[0].imshow(face_image)
    axs[0].axis("off")
    for i in range(len(hmm['pdf'])):
        mean = (hmm['pdf'][i]['mean'].flatten() * np.array([face_width / screen_resolution[0], face_height / screen_resolution[1]]))
        #cov = hmm['pdf'][i]['cov']
        cov = hmm['pdf'][i]['cov'] * np.array([face_width / screen_resolution[0], face_height / screen_resolution[1]])**2
        v, w = np.linalg.eigh(cov)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        
        color = colors[i % len(colors)] 
        ell = plt.matplotlib.patches.Ellipse(mean, v[0], v[1],180.0, facecolor=color)
        ell.set_clip_box(axs[0].bbox)
        ell.set_alpha(0.5)
        axs[0].add_artist(ell)

    #transition matrix
    im = axs[1].imshow(hmm['trans'], interpolation='nearest', cmap='Blues')
    axs[1].set_title('Matrice di transizione')
    plt.colorbar(im, ax=axs[1])

    # Prior
    axs[2].bar(np.arange(3), hmm['prior'].flatten())
    axs[2].set_title('Prior')
    axs[2].set_xlabel('Stato')
    axs[2].set_ylabel('Probabilita')

    # Mostra la figura
    plt.tight_layout()
    plt.show()


