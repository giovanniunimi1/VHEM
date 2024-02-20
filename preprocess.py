import numpy as np
import re
def preprocess_file(file_name):
    scanpath=[]
    fixations = []
    saccades = []
    with open(file_name, "r") as f:
        events = f.readlines()
        #Qui aggiungere funzione per fare il select
        #Ad esempio : 
        # Trova l'indice della prima occorrenza di "TRIALID 32"
        starting_trial = next((i for i, event in enumerate(events) if "TRIALID 100" in event), None)
        ending_trial = next((i for i, event in enumerate(events) if "TRIALID 120" in event), None)
        if starting_trial is not None:
        # Rimuovi tutte le righe precedenti a "TRIALID 32"
                events = events[starting_trial:ending_trial]

        events = [event for event in events if  'MSG' not in event and 'SBLINK' not in event and 'EBLINK' not in event]
        ef_index = np.where(["EFIX" in ev for ev in events])[0]
        sf_index = np.zeros_like(ef_index)  # Creiamo un array vuoto delle stesse dimensioni di ef_index
        for i, ef_idx in enumerate(ef_index):
            ef_line = events[ef_idx]
            ef_values = ef_line.split()
            duration = int(ef_values[3]) - int(ef_values[2])
            sf_index[i] = ef_idx - duration
        ss_index = np.where(["SSAC" in ev for ev in events])[0]
        es_index = np.where(["ESAC" in ev for ev in events])[0]
    fixations=extract_fs(sf_index,ef_index,events,1)
    #saccades=extract_fs(ss_index,es_index,events,0)
    scanpath= fixations #+ saccades 
    scanpath.sort(key=lambda data:data[0])
    scanpath=[data[1] for data in scanpath]
    return scanpath


def pixels2angles(xs, dist, w, h, resx, resy):
    # xs: Nx2 data
    # dist: distance from screen in meters
    # w, h: width and height of the sceen (m)
    # resx, resy: resoluion of the screen

    screenmeters = xs / np.array([resx, resy])  # procedura standard
    screenmeters -= 0.5  # per avere 0,0 al centro dello schermo
    screenmeters *= np.array([w, h])
    angles = np.degrees(np.arctan(screenmeters / dist))
    return angles
def extract_fs(s_index,e_index,events,tipo):
    fs=[]
    for i in range(len(s_index)):
        if tipo:
            start=s_index[i]-1
            end=e_index[i]
        else:
            start=s_index[i]+1
            end=e_index[i]-1
        current_fixation=events[start:end]
        first_iteration = True
        trial_coor=[]
        for event in current_fixation:
            if first_iteration and not event.startswith("SBLINK"):
                z = float(event.split('\t')[0])
                first_iteration=False 
            else :
                try:
                    x = float(event.split('\t')[1])
                    y = float(event.split('\t')[2])                
                    trial_coor.append([x,y])
                except Exception as e:
                    print("INSIDE EXCEPT!")
                    print(e)
        fs.append([z,trial_coor])
    return fs