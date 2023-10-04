from UpdatedLeNet import UpdatedLeNet
from MNIST_Generator import MNIST_Generator
from MNIST_Classifier import MNIST_Classifier

from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import Image, ImageTk, ImageOps
import io
import torch
from torchvision import transforms as T
from tkinter import filedialog
from matplotlib import pyplot as plt
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F



#erosione e dilatazione
from scipy.ndimage import binary_erosion, binary_dilation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False
to_pil = transforms.ToPILImage()

labels_map = {
    0: "Greater",
    1: "Smaller",
    2: "Equal"
}

def visualize_digit_pair(image, print=True):
    # Create a figure and a single subplot
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    # Plot the first digit image
    ax[0].imshow(to_pil(image[0]))
    ax[0].axis('off')

    # Plot the second digit image
    ax[1].imshow(to_pil(image[1]))
    ax[1].axis('off')

    # Print the relation label
    plt.title(f'Relation')

    # Display the plot
    if print:
        plt.show()
    return fig

W, H = 28, 28
MEAN, STD = 0.5, 0.5
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((MEAN,), (STD,))  # Normalize pixel values to range [-1, 1]
])

onehot_before_cod = torch.LongTensor([i for i in range(10)]).to(device) #0123456789
onehot = nn.functional.one_hot(onehot_before_cod, num_classes=10)
onehot = onehot.reshape(10,10,1,1).float()

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):

        #riferimenti dell'immagine
        self.c1_image_tk = None
        self.c2_image_tk = None
        
        self.device = device
        #UpdatedLeNet
        self.model = UpdatedLeNet(n_feature = 6,output_size = 3).to(self.device) #ResNet56
        checkpoint = torch.load("Adam_0_002_ebrmnist_best.pth", map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        #GAN
        self.generator = MNIST_Generator().to(self.device) #DCGAN GENERATOR
        self.generator.load_state_dict(torch.load("./generator_cDCGAN_22.pth", map_location=torch.device(self.device)))
        self.generator.eval()
        #CLASSIFIER
        self.classifier = MNIST_Classifier().to(self.device)
        self.classifier.load_state_dict(torch.load("./classifier_trained.pth", map_location=torch.device(self.device)))
        self.classifier.eval()

        #Canvas
        self.root = Tk()
        self.c1_tensor = None
        self.c2_tensor = None

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.choose_size_button = Scale(self.root, from_=35, to=80, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)
        self.choose_size_button.set(10)

        self.save_button = Button(self.root, text='save', command=self.save_image)
        self.save_button.grid(row=0, column=5)

        self.clear_button1 = Button(self.root, text='clear', command=self.clear_canvas1)
        self.clear_button1.grid(row=2, column=0)

        self.clear_button2 = Button(self.root, text='clear', command=self.clear_canvas2)
        self.clear_button2.grid(row=2, column=3)

        self.label = Label(self.root, text="Draw a picture or upload one", font=("Helvetica", 30))
        self.label.grid(row=5, column=0, columnspan=6)

        self.load_image_button1 = Button(self.root, text='load image', command=self.load_image1)
        self.load_image_button1.grid(row=2, column=1)

        # Create a variable to store the selected number
        self.selected_number1 = IntVar(self.root)
        self.selected_number1.set(0)  # Set the default value to 0

        # Create the dropdown menu with numbers from 0 to 9
        self.number_dropdown1 = OptionMenu(self.root, self.selected_number1, *range(10))
        self.number_dropdown1.grid(row=3, column=0)  # Place it next to the label

        self.generate_button1 = Button(self.root, text='GENERATE', command=self.generate_image1)
        self.generate_button1.grid(row=3, column=1)

        self.load_image_button2 = Button(self.root, text='load image', command=self.load_image2)
        self.load_image_button2.grid(row=2, column=4)

        # Create a variable to store the selected number
        self.selected_number2 = IntVar(self.root)
        self.selected_number2.set(0)  # Set the default value to 0

        # Create the dropdown menu with numbers from 0 to 9
        self.number_dropdown2 = OptionMenu(self.root, self.selected_number2, *range(10))
        self.number_dropdown2.grid(row=3, column=3)  # Place it next to the label

        self.generate_button2 = Button(self.root, text='GENERATE', command=self.generate_image2)
        self.generate_button2.grid(row=3, column=4)


        # Classifier canv1
        self.label_classify1 = Label(self.root, text="_", font=("Helvetica", 12))
        self.label_classify1.grid(row=4, column=1)  # Place it next to the label
        self.classify_button1 = Button(self.root, text='PREDICT', command=self.classify_image1)
        self.classify_button1.grid(row=4, column=0)
        # Classifier canv2
        self.label_classify2 = Label(self.root, text="_", font=("Helvetica", 12))
        self.label_classify2.grid(row=4, column=4)  # Place it next to the label
        self.classify_button2 = Button(self.root, text='PREDICT', command=self.classify_image2)
        self.classify_button2.grid(row=4, column=3)


        self.compare_button = Button(self.root, text='COMPARE', command=self.compare)
        self.compare_button.grid(row=6, column=0, columnspan=6)

        self.c1 = Canvas(self.root, bg='white', width=512, height=512)
        self.c1.grid(row=1, column=0, columnspan=2)

        self.c2 = Canvas(self.root, bg='white', width=512, height=512)
        self.c2.grid(row=1, column=3, columnspan=2)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c1.bind('<B1-Motion>', self.paint)
        self.c1.bind('<ButtonRelease-1>', self.reset)
        self.c2.bind('<B1-Motion>', self.paint)
        self.c2.bind('<ButtonRelease-1>', self.reset)

    def clear_canvas1(self):
        self.c1.delete("all")

    def clear_canvas2(self):
        self.c2.delete("all")

    def use_pen(self):
        self.activate_button(self.pen_button)

    def generate_image1(self):
        
        n = self.selected_number1.get()
        fixed_noise = torch.randn(1,100,1,1)
        generated_image = self.generator(fixed_noise.to(self.device), onehot[n].to(self.device))

        # Crea una maschera booleana dove True rappresenta i pixel che superano la soglia
        maschera = generated_image > 0
        # Azzerare i pixel sotto la soglia impostando i valori corrispondenti a False in maschera a 0
        generated_image[~maschera] = 0

        # Converti il tensore in un array NumPy
        generated_image = generated_image.squeeze().cpu().detach().numpy()

        # Scala i valori nel range (0, 1) se necessario (ad esempio, se i valori del tensore sono normalizzati tra -1 e 1)
        # numpy_image = (numpy_image - numpy_image.min()) / (numpy_image.max() - numpy_image.min())
        # Moltiplica i valori per 255 e convertili in interi (0-255) se i valori del tensore sono in [0, 1]
        generated_image = (generated_image * 255).astype('uint8')
        # Crea un'immagine PIL
        image = Image.fromarray(generated_image)

        #invert color
        image = ImageOps.invert(image)

        # Resize image
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        image_tk = ImageTk.PhotoImage(image)

        # Save pointer of immage to avoid deleting from garbage collector
        self.c1_image_tk = image_tk

        # Crea l'immagine nel canvas
        self.c1.create_image(0, 0, anchor='nw', image=image_tk)
    
    def generate_image2(self):
        n = self.selected_number2.get()
        fixed_noise = torch.randn(1,100,1,1)
        generated_image = self.generator(fixed_noise.to(self.device), onehot[n].to(self.device))

        # Crea una maschera booleana dove True rappresenta i pixel che superano la soglia
        maschera = generated_image > 0
        # Azzerare i pixel sotto la soglia impostando i valori corrispondenti a False in maschera a 0
        generated_image[~maschera] = 0

        # Converti il tensore in un array NumPy
        generated_image = generated_image.squeeze().cpu().detach().numpy()

        # Scala i valori nel range (0, 1) se necessario (ad esempio, se i valori del tensore sono normalizzati tra -1 e 1)
        # numpy_image = (numpy_image - numpy_image.min()) / (numpy_image.max() - numpy_image.min())
        # Moltiplica i valori per 255 e convertili in interi (0-255) se i valori del tensore sono in [0, 1]
        generated_image = (generated_image * 255).astype('uint8')
        # Crea un'immagine PIL
        image = Image.fromarray(generated_image)

        #invert color
        image = ImageOps.invert(image)

        # Resize image
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        image_tk = ImageTk.PhotoImage(image)

        # Save pointer of immage to avoid deleting from garbage collector
        self.c2_image_tk = image_tk

        # Crea l'immagine nel canvas
        self.c2.create_image(0, 0, anchor='nw', image=image_tk)
    
    def load_image1(self):
        self.load_image(self.c1)

    def load_image2(self):
        self.load_image(self.c2)

    def load_image(self, canvas):
            image_path = filedialog.askopenfilename(filetypes=[('All Files', '*.*')])
            if image_path:
                print(image_path)

                # Resize image
                image = Image.open(image_path)
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
                image_tk = ImageTk.PhotoImage(image)

                # Save pointer of immage to avoid deleting from garbage collector
                if canvas == self.c1:
                    self.c1_image_tk = image_tk
                else:  # canvas == self.c2
                    self.c2_image_tk = image_tk

                # Crea l'immagine nel canvas
                canvas.create_image(0, 0, anchor='nw', image=image_tk)

    def classify_image1(self):
        c1_image = self.convert_to_image(self.c1)
        #scala di grigi
        c1_image = c1_image.convert("L")
        #inverte i colori:
        c1_image = ImageOps.invert(c1_image)
        c1_image = c1_image.resize((H, W), Image.Resampling.LANCZOS)

        c1_ten = transform(c1_image).unsqueeze(0).to(self.device)
        print(c1_ten.shape)
        outputs = self.classifier(c1_ten)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)

        '''
        output = self.model(image_concat)

        _, predicted = torch.max(output.data, 1)
        sm = nn.Softmax(dim = 1)
        probs = sm(output)*100.0

        if DEBUG:
            print(predicted)
        '''
        #final_label = f"{labels_map[predicted.item()]}\n{probs[0][predicted].item()}%"

        #self.label.config(text=final_label)
        
    def classify_image2(self):
        c2_image = self.convert_to_image(self.c2)
        #scala di grigi
        c2_image = c2_image.convert("L")
        #inverte i colori:
        c2_image = ImageOps.invert(c2_image)
        c2_image = c2_image.resize((H, W), Image.Resampling.LANCZOS)

        c2_ten = transform(c2_image).unsqueeze(0).to(self.device)
        print(c2_ten.shape)
        outputs = self.classifier(c2_ten)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)

        '''
        output = self.model(image_concat)

        _, predicted = torch.max(output.data, 1)
        sm = nn.Softmax(dim = 1)
        probs = sm(output)*100.0

        if DEBUG:
            print(predicted)
        '''
        #final_label = f"{labels_map[predicted.item()]}\n{probs[0][predicted].item()}%"

        #self.label.config(text=final_label)

    def compare(self):
        c1_image = self.convert_to_image(self.c1)
        c2_image = self.convert_to_image(self.c2)

        #scala di grigi
        c1_image = c1_image.convert("L")
        c2_image = c2_image.convert("L")

        #inverte i colori:
        c1_image = ImageOps.invert(c1_image)
        c2_image = ImageOps.invert(c2_image)

        c1_image = c1_image.resize((H, W), Image.Resampling.LANCZOS)
        c2_image = c2_image.resize((H, W), Image.Resampling.LANCZOS)

        self.c1_tensor = transform(c1_image)
        self.c2_tensor = transform(c2_image)

        image_concat = torch.cat([self.c1_tensor, self.c2_tensor], dim=0) 

        if DEBUG:
            visualize_digit_pair(image_concat)     

        image_concat = image_concat.unsqueeze(0)
        image_concat = image_concat.to(self.device)
        output = self.model(image_concat)

        _, predicted = torch.max(output.data, 1)
        sm = nn.Softmax(dim = 1)
        probs = sm(output)*100.0

        if DEBUG:
            print(predicted)
        
        final_label = f"{labels_map[predicted.item()]}\n{probs[0][predicted].item()}%"

        self.label.config(text=final_label)

    def save_image(self):
        transform = T.ToTensor()

        c1_image = self.convert_to_image(self.c1)
        c2_image = self.convert_to_image(self.c2)

        c1_image = c1_image.resize((H, W), Image.Resampling.LANCZOS)
        c2_image = c2_image.resize((H, W), Image.Resampling.LANCZOS)

        self.c1_tensor = transform(c1_image)
        self.c2_tensor = transform(c2_image)

        if DEBUG:
            print(self.c1_tensor.shape)

        c1_image.save("input1.png")
        c2_image.save("input2.png")
        print("saved")
        
    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            event.widget.create_line(self.old_x, self.old_y, event.x, event.y,
                                     width=self.line_width, fill=paint_color,
                                     capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def convert_to_image(self, canvas):
        postscript = canvas.postscript(colormode='color')
        image = Image.open(io.BytesIO(postscript.encode('utf-8')))
        return image

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    paint_app = Paint()
