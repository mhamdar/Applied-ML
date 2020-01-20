import numpy as np
from skimage.measure import label, regionprops

from skimage.transform import resize

from skimage.transform import *
from random import randrange


def digit_extractor(source, labelled_clusters, top_clusters=3):
    nb_clusters = np.max(labelled_clusters)

    cluster_list = []
    for i in range(1, nb_clusters + 1): #ignoring background
        cluster_size = np.sum(labelled_clusters == i)
        cluster_list.append((cluster_size, i))  # append tuple

    cluster_list.sort(reverse=True)

    cluster_imgs = []
    for _, cluster in cluster_list[:top_clusters + 1]: #assuming that the background cluster is the largest

        mask = labelled_clusters == cluster
        unmasked = unmask(source, mask)
        cropped_digit = crop(unmasked, mask)
        padded_digit = center_by_bounding_box(cropped_digit)

        cluster_imgs.append(padded_digit)

    return cluster_imgs

def unmask(source, mask):
    return source * mask

def crop(source, mask):
    width = source.shape[0]
    height = source.shape[1]

    h_presence = np.sum(mask, axis=0)
    v_presence = np.sum(mask, axis=1)

    h_min = min([i for i in range(width) if h_presence[i]])
    h_max = max([i for i in range(width) if h_presence[i]])

    v_min = min([i for i in range(height) if v_presence[i]])
    v_max = max([i for i in range(height) if v_presence[i]])

    return source[v_min:v_max, h_min:h_max]


def center_by_bounding_box(source, box_size=28):
    height = source.shape[0]
    width = source.shape[1]

    cropped_source = source

    if height > box_size:
        excedent = int((height - box_size)/2)
        cropped_source = cropped_source[excedent:excedent+box_size, : ]

        height = box_size

    if width > box_size:
        excedent = int((width - box_size) / 2)
        cropped_source = cropped_source[ : , excedent:excedent + box_size]

        width = box_size

    canvas = np.zeros((box_size, box_size))  # zero array of size box_size

    height_pad = int((box_size - height) / 2)
    width_pad = int((box_size - width) / 2)

    canvas[height_pad:height_pad + height, width_pad:width_pad + width] = cropped_source

    return canvas


def remove_clusters_below_size(img, size=5):
    labelled_img, nb_clusters = label(img, connectivity=2, return_num=True)
    cluster_ids_to_eliminate = [0]  # 0 is the background cluster

    for i in range(1, nb_clusters):
        cluster_size = np.sum(labelled_img == i)

        if size is not None and cluster_size <= size:
            cluster_ids_to_eliminate.append(i)

    elimination_map = lambda x: 0 if x in cluster_ids_to_eliminate else 255
    vem = np.vectorize(elimination_map)
    declustered_img = vem(labelled_img)
    return declustered_img

def threshold(source, threshold=220):
    f = lambda x : 255 if x >= threshold else 0
    vf = np.vectorize(f)

    return vf(source)

#new methods to get a pristine 28x28 box on which to add a digit. Use data_creator.py

def pristine_box(img):#can produce waaaay more

    bounding_boxes=get_bounding_boxes(img)
    subdiv=get_subdiv(bounding_boxes)
    legal_box_dim=get_intersections(subdiv)

    pristine=[np.array(resize(img[box_dim[0]:box_dim[2], box_dim[1]:box_dim[3]], (128,128))) for box_dim in legal_box_dim]

    return pristine

def tiny_pristine_box(img):#can produce waaaay more

    bounding_boxes=get_bounding_boxes(img)
    subdiv=get_subdiv(bounding_boxes)
    legal_box_dim=[get_max_intersections(subdiv)]

    pristine=[np.array(resize(img[box_dim[0]:box_dim[2], box_dim[1]:box_dim[3]], (128,128))) for box_dim in legal_box_dim]

    return pristine

def get_bounding_boxes(img):

    threshold_img=threshold(img)
    labelled_temp, nb_clusters = label(threshold_img, connectivity=2, return_num=True) #could avoid recalling this
    props=regionprops(labelled_temp)
    bounding_boxes=[x.bbox for x in props]

    return bounding_boxes

def get_subdiv(bounding_boxes):
    subdiv=[]
    size=128
    for x in bounding_boxes:
        top=(0, 0, x[0], size)
        right=(0, x[3], size, size)
        bot=(x[2], 0, size, size)
        left=(0, 0, size, x[1])
        subdiv.append((top, right, bot, left))
    
    return subdiv


def get_intersections(subdiv, thresh=28*28): #assuming we have 3 clusters
    legal=[]
    for x in subdiv[0]:
        for y in subdiv[1]:
            for z in subdiv[2]:
                box=intersection(x, y, z)
                size=(box[2]-box[0])*(box[3]-box[1])
                if size>thresh:
                    legal.append(box)
    return legal

def get_max_intersections(subdiv, thresh=28*28): #assuming we have 3 clusters
    max_size=0
    max_dim=(0,0,0,0)
    for x in subdiv[0]:
        for y in subdiv[1]:
            for z in subdiv[2]:
                box=intersection(x, y, z)
                size=(box[2]-box[0])*(box[3]-box[1])
                if size>max_size:
                    max_size=size
                    max_dim=box
    return max_dim



def intersection(x, y, z):
    (minrow, mincol, maxrow, maxcol)=(max(x[0], y[0], z[0]), max(x[1], y[1], z[1]), min(x[2], y[2], z[2]), min(x[3], y[3], z[3]))
    if minrow>=maxrow or mincol>=maxcol:
        return (0, 0, 0, 0)
    else:
        return (minrow, mincol, maxrow, maxcol)



def reshape_pristine(pristine):
    (width, height)=pristine.shape

    if width<28 or height<28:
        print("kawaboonga")
        return None
    else:
        w_cush=int((width-28)/2)
        h_cush=int((height-28)/2)
        w2=w_cush+28
        h2=h_cush+28
        return pristine[w_cush:w2, h_cush:h2]

threshold2 = np.vectorize(lambda x: x if x < 255 else 255)

def combine_pristine_mnist(pristine, mnist):
    return threshold2(pristine + mnist)

def thruple_mnist(digit1, digit2, digit3, angle1=0, angle2=0, angle3=0):
    img=np.zeros((128, 128))
    
    #digit1=resize(digit1, (28, 28))
    #digit2=resize(digit2, (28, 28))
    #digit3=resize(digit3, (28, 28))
    
    x_1=randrange(0, 100)
    y_1=randrange(0, 100)

    x_2=randrange(0, 100)
    y_2=randrange(0, 100)
    
    while (x_2<x_1+28 and x_2>=x_1-28) and (y_2<y_1+28 and y_2>=y_1-28):
        x_2=randrange(0, 100)
        y_2=randrange(0, 100)

    x_3=randrange(0, 100)
    y_3=randrange(0, 100)

    while ((x_3<x_1+28 and x_3>=x_1-28) and (y_3<y_1+28 and y_3>=y_1-28) or (x_3<x_2+28 and x_3>=x_2-28) and (y_3<y_2+28 and y_3>=y_2-28)):
        x_3=randrange(0, 100)
        y_3=randrange(0, 100)

    img[x_1:x_1+28, y_1:y_1+28]=rotate(digit1, angle1, preserve_range=True)
    img[x_2:x_2+28, y_2:y_2+28]=rotate(digit2, angle2, preserve_range=True)
    img[x_3:x_3+28, y_3:y_3+28]=rotate(digit3, angle3, preserve_range=True)

    return img
