from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, session
import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from deepface import DeepFace
import random
from time import sleep
from datetime import datetime
import logging
from fpdf import FPDF
import plotly.express as px
import pandas as pd
import base64
from io import BytesIO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'b3777ab1c5aac43bc65ea6613200753671fea5126c200dcb'

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
REPORT_FOLDER = 'static/reports'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Language support
LANGUAGES = {
    "en": {
        "menu_title": "🌟 Face and Image Analysis System 🌟",
        "camera_analysis": "🎥 Start Face Analysis from Camera",
        "image_analysis": "📷 Analyze Images from Dataset",
        "upload_analysis": "📤 Upload and Analyze Images",
        "exit": "🚪 Exit",
        "select_option": "Select an option:",
        "starting_camera": "Starting camera...",
        "camera_failed": "Failed to open camera.",
        "camera_active": "Camera is active. Press 'q' to quit.",
        "analysis_failed": "Analysis failed: {}",
        "total_images": "🖼️ Total images analyzed: {}",
        "real_images": "✅ Real images: {}",
        "fake_images": "❌ Fake images: {}",
        "emotion_distribution": "===== Emotion Distribution =====",
        "report_generated": "Report generated: {}",
        "upload_images": "Upload Images for Analysis",
        "analyze": "Analyze",
        "back": "Back",
        "results": "Analysis Results",
        "emotion": "Emotion",
        "download_report": "Download Report",
        "no_file": "No file selected",
        "file_uploaded": "File uploaded successfully",
        "analyzing": "Analyzing...",
        "real": "Real",
        "fake": "Fake",
        "home": "Home",
    },
    "tr": {
        "menu_title": "🌟 Yüz ve Görüntü Analizi Sistemi 🌟",
        "camera_analysis": "🎥 Kameradan Yüz Analizi Başlat",
        "image_analysis": "📷 Veri Setinden Görüntü Analizi Yap",
        "upload_analysis": "📤 Görüntü Yükle ve Analiz Et",
        "exit": "❌ Çıkış",
        "select_option": "Seçiminizi yapın:",
        "starting_camera": "Kamerayı başlatıyor...",
        "camera_failed": "Kamera açılamadı.",
        "camera_active": "Kamera aktif. Çıkmak için 'q' tuşuna basın.",
        "analysis_failed": "Analiz başarısız: {}",
        "total_images": "🖼️ Analiz edilen toplam görüntü sayısı: {}",
        "real_images": "✅ Gerçek görüntü sayısı: {}",
        "fake_images": "❌ Sahte görüntü sayısı: {}",
        "emotion_distribution": "===== Duyguların Dağılımı =====",
        "report_generated": "Rapor oluşturuldu: {}",
        "upload_images": "Analiz için Görüntü Yükle",
        "analyze": "Analiz Et",
        "back": "Geri",
        "results": "Analiz Sonuçları",
        "emotion": "Duygu",
        "download_report": "Raporu İndir",
        "no_file": "Dosya seçilmedi",
        "file_uploaded": "Dosya başarıyla yüklendi",
        "analyzing": "Analiz ediliyor...",
        "real": "Gerçek",
        "fake": "Sahte",
        "home": "Ana Sayfa",
    }
}

# Default language
LANGUAGE = "tr"

def t(key, *args):
    """Translate text based on selected language with safe formatting."""
    translation = LANGUAGES[LANGUAGE].get(key, key)
    
    # إذا كان النص يحتوي على علامات تنسيق وكانت هناك معطيات
    if args and ('{' in translation and '}' in translation):
        try:
            return translation.format(*args)
        except (IndexError, KeyError):
            return translation
    return translation

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_pdf_report(emotions_count, source, real_images_count=None, fake_images_count=None, image_paths=None):
    """Generate PDF report from analysis results."""
    pdf = FPDF()
    pdf.add_page()
    
    # استخدام الخطوط المدمجة (لا تحتاج لملفات خارجية)
    pdf.set_font("helvetica", "", 12)
    
    # عنوان التقرير
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(200, 10, txt="==== Yuz Analiz Raporu ====", ln=True, align='C')
    pdf.set_font("helvetica", "", 12)
    pdf.cell(200, 10, txt=f"Rapor Tarihi: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)
    
    # قسم مصدر التحليل
    pdf.set_font("helvetica", "B", 12)
    if source == "camera":
        pdf.cell(200, 10, txt="1. Kameradan Yuz Analizi", ln=True)
    elif source == "image_analysis":
        pdf.cell(200, 10, txt="1. Veri Seti Analizi", ln=True)
    elif source == "upload":
        pdf.cell(200, 10, txt="1. Yuklenen Goruntulerin Analizi", ln=True)
    
    pdf.set_font("helvetica", "", 12)
    if source in ["image_analysis", "upload"]:
        pdf.cell(200, 10, txt=f"Toplam Goruntu Sayisi: {real_images_count + fake_images_count}", ln=True)
        pdf.cell(200, 10, txt=f"Gercek Goruntuler: {real_images_count}", ln=True)
        pdf.cell(200, 10, txt=f"Sahte Goruntuler: {fake_images_count}", ln=True)
    pdf.ln(10)
    
    # قسم توزيع المشاعر
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(200, 10, txt="2. Duygu Dagilimi", ln=True)
    pdf.set_font("helvetica", "", 12)
    
    for emotion, count in emotions_count.items():
        pdf.cell(200, 10, txt=f"{emotion.capitalize()}: {count}", ln=True)
    pdf.ln(10)
    
    # إضافة الصور المحللة إذا وجدت
    if image_paths and len(image_paths) > 0:
        pdf.add_page()
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(200, 10, txt="3. Analiz Edilen Goruntuler", ln=True)
        pdf.set_font("helvetica", "", 12)
        
        for i, img_path in enumerate(image_paths):
            if i % 2 == 0 and i != 0:
                pdf.add_page()
            
            try:
                if os.path.exists(img_path):
                    pdf.image(img_path, x=10, y=pdf.get_y(), w=90)
                    pdf.cell(200, 5, txt=f"Goruntu: {os.path.basename(img_path)}", ln=True)
                    pdf.ln(85)
                else:
                    logging.warning(f"Resim bulunamadi: {img_path}")
            except Exception as e:
                logging.error(f"PDF'ye resim eklenirken hata: {str(e)}")
                continue
    
    # حفظ التقرير
    report_name = f"rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_path = os.path.join(app.config['REPORT_FOLDER'], report_name)
    
    try:
        pdf.output(report_path)
        logging.info(f"Rapor basariyla olusturuldu: {report_path}")
    except Exception as e:
        logging.error(f"Rapor kaydedilirken hata: {str(e)}")
        return None
    
    return report_name

def analyze_uploaded_images(file_paths):
    """Analyze uploaded images and return results."""
    results = []
    emotions_count = {'happy': 0, 'sad': 0, 'angry': 0, 'surprise': 0, 'fear': 0, 'neutral': 0}
    
    for file_path in file_paths:
        try:
            # Load and preprocess image
            img = load_img(file_path, target_size=(170, 170))
            img_array = img_to_array(img) / 255.0
            
            # Fake/real detection (simplified - replace with your actual model)
            is_real = random.random() > 0.5  # Replace with your model prediction
            
            # Face analysis
            rgb_image = cv2.cvtColor((img_array * 255).astype("uint8"), cv2.COLOR_BGR2RGB)
            analysis = DeepFace.analyze(rgb_image, actions=['emotion'], enforce_detection=False)
            
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            emotion = analysis.get('dominant_emotion', "unknown")

            
            emotions_count[emotion] += 1
            
            # Save visualization
            plt.figure(figsize=(8, 6))
            plt.imshow(img_array)
            plt.axis('off')
            plt.title(f"{'Gerçek' if is_real else 'Sahte'}", fontsize=16, color='darkgreen')
            plt.gca().text(0.5, -0.1, f"Duygu: {emotion}", ha='center', va='top', fontsize=14, color='blue', transform=plt.gca().transAxes)
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close()
            img_buffer.seek(0)
            img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
            
            results.append({
                'filename': os.path.basename(file_path),
                'is_real': is_real,
                'emotion': emotion,
            })
            
        except Exception as e:
            logging.error(f"Error analyzing image {file_path}: {str(e)}")
            continue
    
    return results, emotions_count

@app.route('/')
def home():
    return render_template('index.html', t=t, language=LANGUAGE)

@app.route('/set_language/<lang>')
def set_language(lang):
    global LANGUAGE
    if lang in LANGUAGES:
        LANGUAGE = lang
    return redirect(request.referrer or url_for('home'))

@app.route('/camera')
def camera():
    return render_template('camera.html', t=t, language=LANGUAGE)

@app.route('/analyze_dataset')

def analyze_dataset():
    # Generate random counts for real and fake, ensuring the total count is at least 30
    real_count = random.randint(10, 20)
    fake_count = random.randint(10, 20) 
    
    # Ensure the sum of real_count and fake_count is at least 30
    while real_count + fake_count < 30:
        fake_count = random.randint(10, 20)
    
    emotions_count = {'happy': random.randint(5, 15), 
                      'sad': random.randint(0, 5), 
                      'angry': random.randint(0, 5), 
                      'surprise': random.randint(0, 5), 
                      'fear': random.randint(0, 5), 
                      'neutral': random.randint(5, 10)}

    # Generate charts
    fig1 = px.bar(x=list(emotions_count.keys()), y=list(emotions_count.values()), 
                  labels={'x': t('emotion'), 'y': 'Count'}, 
                  title=t('emotion_distribution'), 
                  color=list(emotions_count.keys()))
    
    fig2 = px.pie(names=[t('real'), t('fake')], 
                 values=[real_count, fake_count], 
                 title=f"{t('real')} vs {t('fake')}",
                 color=[t('real'), t('fake')],
                 color_discrete_map={t('real'): 'green', t('fake'): 'red'})
    
    # Convert charts to HTML
    chart1_html = fig1.to_html(full_html=False)
    chart2_html = fig2.to_html(full_html=False)
    
    # Generate report
    report_name = generate_pdf_report(emotions_count, "image_analysis", real_count, fake_count)
    
    return render_template('dataset_results.html', 
                         t=t, 
                         language=LANGUAGE,
                         emotions_count=emotions_count,
                         real_count=real_count,
                         fake_count=fake_count,
                         chart1_html=chart1_html,
                         chart2_html=chart2_html,
                         report_name=report_name)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if files were uploaded
        if 'files[]' not in request.files:
            return jsonify({'error': t('no_file')})
        
        files = request.files.getlist('files[]')
        file_paths = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                file_paths.append(file_path)
        
        if not file_paths:
            return jsonify({'error': t('no_file')})
        
        # Store file paths in session for analysis
        session['file_paths'] = file_paths
        return jsonify({'message': t('file_uploaded'), 'redirect': url_for('analyze_uploaded')})
    
    return render_template('upload.html', t=t, language=LANGUAGE)

@app.route('/analyze_uploaded')
def analyze_uploaded():
    file_paths = session.get('file_paths', [])
    if not file_paths:
        return redirect(url_for('upload'))
    
    results, emotions_count = analyze_uploaded_images(file_paths)
    
    # Generate charts
    fig1 = px.bar(x=list(emotions_count.keys()), y=list(emotions_count.values()), 
                  labels={'x': t('emotion'), 'y': 'Count'}, 
                  title=t('emotion_distribution'), 
                  color=list(emotions_count.keys()))
    
    real_count = sum(1 for r in results if r['is_real'])
    fake_count = len(results) - real_count
    
    if len(results) > 0:
        fig2 = px.pie(names=[t('real'), t('fake')], 
                     values=[real_count, fake_count], 
                     title=f"{t('real')} vs {t('fake')}",
                     color=[t('real'), t('fake')],
                     color_discrete_map={t('real'): 'green', t('fake'): 'red'})
        chart2_html = fig2.to_html(full_html=False)
    else:
        chart2_html = None
    
    # Generate report
    report_name = generate_pdf_report(emotions_count, "upload", real_count, fake_count, file_paths)
    
    return render_template('upload_results.html', 
                         t=t, 
                         language=LANGUAGE,
                         results=results,
                         emotions_count=emotions_count,
                         real_count=real_count,
                         fake_count=fake_count,
                         chart1_html=fig1.to_html(full_html=False),
                         chart2_html=chart2_html,
                         report_name=report_name)

@app.route('/download_report/<filename>')
def download_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

@app.route('/capture_image', methods=['POST'])
def capture_image():
    # In a real implementation, this would capture an image from the webcam
    # For this example, we'll simulate it with a random image from the dataset
    
    # Simulate capturing an image
    data_dir = r'.\Data'  # Replace with your actual dataset path
    real_data = [f for f in os.listdir(os.path.join(data_dir, 'training_real')) if f.endswith('.jpg')]
    
    if not real_data:
        return jsonify({'error': 'No images found in dataset'})
    
    # Select a random image
    img_name = random.choice(real_data)
    img_path = os.path.join(data_dir, 'training_real', img_name)
    
    # Copy to uploads folder with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_filename = f"captured_{timestamp}.jpg"
    new_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    
    # Read and save the image
    img = cv2.imread(img_path)
    cv2.imwrite(new_path, img)
    
    # Analyze the image
    try:
        analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        emotion = analysis.get('dominant_emotion', "unknown")
       
        
        # Create visualization
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Duygu: {emotion}", fontsize=16, color='darkgreen')
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image_data': img_data,
            'analysis': {
                'emotion': emotion,

            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Server stopped by user")
        print('==========================')
    print(' URL: http://localhost:5000/')
    print('Açmak Için CTRL+CLİCK ')
    print('==========================')


