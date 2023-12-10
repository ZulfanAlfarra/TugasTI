from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
# from PIL import Image


model = load_model('model_tl_inception.h5')
classes = ['bagong', 'cepot', 'gareng', 'petruk', 'semar']
information = ['Wayang Bagong adalah salah satu karakter punakawan dalam seni pertunjukan tradisional Jawa, seperti wayang kulit. Bagong adalah tokoh yang lucu dan ceria, sering memberikan humor dalam cerita wayang kulit dan juga pesan moral. Karakternya memiliki wajah bulat dengan moncong panjang, dan ia merupakan tokoh yang sangat disukai dalam pertunjukan wayang kulit.',
               'Wayang Cepot adalah salah satu tokoh punakawan dalam seni pertunjukan tradisional Sunda, khususnya dalam pertunjukan wayang golek Sunda. Wayang golek adalah seni pertunjukan tradisional dari Jawa Barat, Indonesia, yang menggunakan boneka kayu untuk menggambarkan berbagai tokoh dalam cerita. Cepot adalah tokoh punakawan yang cerdas, ceria, dan sering kali menjadi sumber humor dalam cerita wayang golek Sunda. Ia dikenal dengan tingkah laku kocaknya dan sering memberikan nasihat atau petuah dalam bentuk humor kepada tokoh-tokoh lain dalam pertunjukan. Wayang Cepot adalah salah satu tokoh yang paling populer dalam wayang golek Sunda dan merupakan bagian penting dari seni dan budaya Sunda.',
               'Wayang Gareng adalah salah satu tokoh punakawan dalam seni pertunjukan wayang kulit Jawa yang cerdas, berwawasan, dan sering memberikan nasihat bijak dalam cerita. Dengan wajah bulat, moncong pendek, dan rambut yang kusut, Gareng tidak hanya memberikan humor, tetapi juga memiliki karakter serius yang mendalam, sering berperan sebagai penasihat raja atau tokoh utama. Ia memainkan peran penting dalam menjaga keseimbangan karakter punakawan dan memberikan pesan moral dalam pertunjukan wayang kulit, menjadikannya salah satu tokoh yang sangat dihormati dan disukai dalam budaya Jawa.',
               'Wayang Petruk adalah salah satu tokoh punakawan dalam seni pertunjukan wayang kulit Jawa yang memiliki wajah panjang, moncong panjang, rambut kuncir, dan penampilan kocak. Ia seringkali dihadirkan sebagai tokoh yang ceroboh, naif, atau konyol dalam cerita, menyediakan elemen humor dan komedi dalam pertunjukan. Petruk juga berperan sebagai teman atau mitra tokoh utama atau raja dalam cerita dan terkadang memberikan nasihat, meskipun seringkali dalam bentuk yang konyol. Selain memberikan hiburan, Petruk juga berperan dalam menjaga keseimbangan antara karakter punakawan lainnya dalam pertunjukan wayang kulit, menjadikannya salah satu tokoh yang sangat disukai oleh penonton dan merupakan bagian penting dari budaya Jawa.',
               'Wayang Semar adalah salah satu tokoh punakawan yang sangat terkenal dalam seni pertunjukan tradisional Jawa, terutama dalam pertunjukan wayang kulit. Semar adalah tokoh yang khas dengan penampilan yang gemuk, tiga moncong, dan wajah tersenyum yang ramah. Ia sering dianggap sebagai tokoh tertua dan bijak dalam kelompok punakawan, berperan sebagai pelawak yang memberikan humor, nasihat, serta petuah yang dalam dalam cerita wayang. Selain sebagai sumber hiburan, Semar juga seringkali dianggap sebagai penjaga dan pelindung dalam pertunjukan wayang, dan memiliki nilai simbolis yang mendalam dalam budaya Jawa, menggambarkan kebijaksanaan dan harmoni dalam kehidupan.']

def predict_label(img_path):
    img = image.load_img(img_path, target_size=(150,150))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    output = model.predict(images)
    best_index = np.argmax(output)

    return classes[best_index], information[best_index]


app = Flask(__name__, template_folder='views')

# routes
@app.route("/", methods = ['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)
    
    return render_template("index.html", prediction=p, img_path= img_path)

if __name__ == '__main__':
    app.run(debug=True)