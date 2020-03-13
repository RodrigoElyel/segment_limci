import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, show, figure, imsave as pltsave, stackplot
from skimage.io import imread, imsave
from skimage.util import img_as_float, invert, img_as_ubyte
from skimage.color import rgb2gray, rgb2hsv, label2rgb
from skimage.filters import threshold_yen, threshold_otsu,median, edges
from skimage.exposure import equalize_hist
from skimage.morphology import binary_closing, binary_opening, binary_erosion, binary_dilation, closing
from skimage.morphology.selem import disk, square
from skimage.segmentation import slic, slic_superpixels, mark_boundaries, clear_border
from skimage.measure import regionprops, label
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from scipy.spatial import distance
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics

origin_dir = "./img/"
dest_dir = "./results/"

for file in list(os.scandir(origin_dir))[2:3]:
    input_img = img_as_float(imread(file.path))

    #ESPAÇOS DE CORES E MEDIANA
    img = equalize_hist(input_img)[:,:,0] # enhance Contrast
    img2 = equalize_hist(input_img)[:,:,1] # enhance Contrast
    img_hsvS = rgb2hsv(input_img)[:,:,1]
    img_median = median(img, selem=disk(7))

    # SUPERPIXELS
    imgTeste = img_as_ubyte(input_img)
    segments_slic = slic(imgTeste, n_segments=4000, compactness=10, sigma=1)
    sp = mark_boundaries(imgTeste, segments_slic)
    figure("super")
    imshow(sp)


    # image_label_overlay = label2rgb(segments_slic, image=input_img)
    # figure("uper")
    # imshow(image_label_overlay)

    region = regionprops(segments_slic, intensity_image=None, cache=True)

    lista = []
    count = 0
    for prop in region:
        if prop.area > 500 and prop.area < 700 :
            lista.append(prop.area)
        else:
            lista.append(0)

    # show()
    # exit()
    # stackk = np.stack((img, img2, img_hsvS, img_median), axis=2)
    # img = np.reshape(stackk, (1080 * 1920, 4))

    kmc = KMeans(n_clusters=8, n_init=1, verbose=False, random_state=0, n_jobs=8)
    mask = kmc.fit_predict(lista)

    # mask = np.reshape(mask, (1080, 1920,))

    figure("Grouping")
    imshow(mask, cmap='Paired')
    pltsave(dest_dir + file.name, mask, cmap='Paired')

    mask = mask == 5 #indice # label for color id

    mask = binary_closing(mask, disk(3))
    mask = binary_opening(mask, disk(3))

    figure("Choose one cluster id")
    imshow(mask, cmap='gray')
    pltsave('./results/{}_mask.png'.format(file.name), mask, cmap='gray')

    figure("Show only detections")
    imshow(np.expand_dims(mask, 2) * input_img, cmap='gray')
    figure("Show only no detections")
    imshow(np.expand_dims(invert(mask), 2) * input_img, cmap='gray')
    pltsave('./results/{}_final.png'.format(file.name), (np.expand_dims(invert(mask), 2) * input_img), cmap='Paired')

    show()


    # for i in range(0, 7):
    #     print("Linha - {}".format(i))
    #     for j in range(0, 4):
    #        print("Coluna - {} -> {}".format(j, kmc.cluster_centers_[i][j]))

    # vet1 = [0.22239735, 0.1193011,  0.37232683, 0.25981909] #3
    # vet2 = [0.08043895, 0.05862604, 0.66156754, 0.11438161] #5
    # vet3 = [0.24279948, 0.13502257, 0.37295382, 0.27008074] #6
    # vet4 = [0.08063511, 0.05267341, 0.29420347, 0.10978902] #3
    # vet5 = [0.06083852, 0.02246094, 0.32080379, 0.09984489] #3
    # vet6 = [0.06749587, 0.02106527, 0.57497418, 0.10935119] #6
    # vet7 = [0.15506686, 0.09801951, 0.39420678, 0.19358028] #1
    # vet8 = [0.16218812, 0.11099203, 0.37143792, 0.19534411] #3
    # vet9 = [0.25061039, 0.20545334, 0.52024826, 0.28420233] #4
    # vet10 = [0.23563074, 0.16774033, 0.43781835, 0.28154468] #6
    #
    # pos0 = (vet1[0] + vet2[0] + vet3[0] + vet4[0] + vet5[0] + vet6[0] + vet7[0] + vet8[0] + vet9[0] + vet10[0])/10
    # pos1 = (vet1[1] + vet2[1] + vet3[1] + vet4[1] + vet5[1] + vet6[1] + vet7[1] + vet8[1] + vet9[1] + vet10[1])/10
    # pos2 = (vet1[2] + vet2[2] + vet3[2] + vet4[2] + vet5[2] + vet6[2] + vet7[2] + vet8[2] + vet9[2] + vet10[2])/10
    # pos3 = (vet1[3] + vet2[3] + vet3[3] + vet4[3] + vet5[3] + vet6[3] + vet7[3] + vet8[3] + vet9[3] + vet10[3])/10
    #
    # newVet = [pos0, pos1, pos2, pos3]
    # newVet = np.reshape(newVet, (4,))
    #
    # menor = 100000
    # indice = -1
    #
    # for i in range(0, 7):
    #     dist = distance.euclidean([newVet], [kmc.cluster_centers_[i]])
    #     if dist < menor:
    #         menor = dist
    #         indice = i

    # LBP
    # radius = 3
    # n_points = 8*radius
    # lbp_default = local_binary_pattern(img, n_points, radius, 'default')
    # lbp_ror = local_binary_pattern(img, n_points, radius, 'ror')
    # lbp_uniform = local_binary_pattern(img, n_points, radius, 'uniform')
    # lbp_nri = local_binary_pattern(img, n_points, radius, 'nri_uniform')
    # lbp_var = local_binary_pattern(img, n_points, radius, 'var')
    # figure("lbp")
    # imshow(lbp, cmap='gray')

    #GLCM
    # imgTeste = img_as_ubyte(img)
    # g = greycomatrix(imgTeste, [50], [0, np.pi/2], levels=256)
    # contrast = greycoprops(g, 'contrast')
    # primeira_glcm = g[:, :, 0, 0]
    # segunda_glcm = g[0, 0, :, :]
    # terceira = g[:, :, 0, 0]
    # quarta_glcm = g[:, :, 0, 0]
    # print(contrast)
    # figure("GLCM")
    # imshow(primeira_glcm, cmap='hot')
    # imgTextura = np.reshape(lbp_default, (1080 * 1920, 1))
    # show()
    # exit()