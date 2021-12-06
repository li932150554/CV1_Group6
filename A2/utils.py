import matplotlib.pyplot as plt

def show_image(im):
    plt.imshow(im, interpolation='bilinear')
    
    
def show_images(ims, hw, size=(8, 2)):
    assert ims.shape[0] < 10, "Too many images to display"
    n = ims.shape[0]
    
    # visualising the result
    fig = plt.figure(figsize=size)
    for i, im in enumerate(ims):
        fig.add_subplot(1, n, i + 1)
        plt.imshow(im.reshape(*hw))