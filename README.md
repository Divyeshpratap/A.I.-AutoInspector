# Automobile Inspector: AI-Powered Product for Car Visual Damage Detection, parametric cost estimation, and car maintenance chatbot

![Generative AI](https://img.shields.io/badge/Generative%20AI-green)
![ChatBot](https://img.shields.io/badge/Chat%20Bot-blue)
![Image Segmentation](https://img.shields.io/badge/Image%20Segmentation-orange)
![Object Detection](https://img.shields.io/badge/Object%20Detection-yellow)
![Mask R-CNN](https://img.shields.io/badge/Mask%20R--CNN-red)
![Deformable ConvNets](https://img.shields.io/badge/DCN-Deformable%20ConvNets-yellowgreen)
![Faster R-CNN](https://img.shields.io/badge/Faster%20R--CNN-purple)
![LangChain](https://img.shields.io/badge/LangChain-blue)
![FAISS](https://img.shields.io/badge/FAISS-white)
![Llama](https://img.shields.io/badge/Llama-grey)
![Flask](https://img.shields.io/badge/Flask-black)


## Overview
Automobile Inspector is an advanced AI-powered application designed to automate vehicle visual **damage assessment** and supporting customers with technical car manuals and their queries via a conversational **chatBot**. The project integrates latest computer vision techniques with generative A.I to provide a comprehensive product for automotive inspections and user inquiries.

## Table of Contents
1. [Description](#description)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Directory](#directory)
5. [Data](#data)
6. [Result](#result)
7. [Acknowledgment](#acknowledgment)
8. [Contact](#Contact)

<div align="center">
  <video src="https://github.com/user-attachments/assets/846b8083-fc78-45d4-9aee-14719e440734.mp4" controls width="75%">
    Your browser does not support the video tag.
  </video>
  <p><i>Visual Damage Detection Demonstration</i></p>
</div>

   
## Description
The system offers two core functionalities:
### 1. Visual Damage Detection: 
Utilizing a Mask R-CNN, the system processes uploaded vehicle images, **categorizes** and **localizes** damages (dents, scratches, cracks, glass shatter, broken lamp and flat tire) leveraging a ResNet-101 backbone with Deformable Convolution Network. Repair costs are estimated using a parameterized equation leveraging area and type of damage (segmentation model), and car cost (scraped from internet using google API). User can then request repair services from nearest agent using the portal. 
<div align="center">
   <img src="https://github.com/user-attachments/assets/1442c5ef-e539-4063-822e-1f51819eb23a" alt="visualDamagePipeline" width="75%">
</div>

### 2. RAG based carBot: 
carBot provides a **conversation style chatbot** that answers user queries about their vehicle. It uses a Retrieval-Augmented Generation (RAG) pipeline that allows users to upload car manuals or provide web links. The system processes the question further refining it by utilizing the chat history, extracts relevant documents from the FAISS vectorstore, and uses a pre-trained large language model to answer user queries about their vehicle. Since the model gives answers based on the user uploaded car manual, this ensures that the information provided is accurate, context-specific, and directly related to the user's specific vehicle model.
<div align="center">
   <img src="https://github.com/user-attachments/assets/3670f274-12a4-45fa-afe1-1c342c6d0334" alt="chatBot Interface" width="50%">   
</div>

## Architecture

The Automobile Inspector system is composed of several key components, each serving a distinct role in the overall architecture:
### 1. Frontend:
- **User Interface:** The application provides an intuitive and user-friendly interface, built with **HTML, CSS, and JavaScript**. Users can upload images, provide car manual PDFs, and interact with the system through a clean and responsive design.
- **carbot Interface:**  A dedicated interface for the NLP-based chatbot allows users to ask questions about their vehicle, with the system retrieving and presenting relevant information from uploaded manuals.

### 2. Backend:
- **Flask Server:** The backend is powered by **Flask**, a lightweight Python web framework. The server handles user requests, manages sessions, processes images, and interacts with the machine learning models.
- **RESTful API:**  The backend exposes several endpoints for image uploads, damage detection, manual processing, and query handling, making the system modular and easy to extend.
- **Database:** Relational **SQLlite** database for storing user, order, predictions, etc. information.

### 3. Computer Vision Module:
- **Image Segmentation-Based Damage Detection:**
- The damage detection pipeline is powered by a **Mask R-CNN** architecture, which leverages a **ResNet-101 backbone** with **Deformable Convolutional Networks (DCN)** to enhance feature extraction, especially for detecting irregularly shaped objects like vehicle damages. The model is fine-tuned to detect and segment six specific damage types: dent, scratch, crack, glass shatter, lamp broken, and tire flat.
- The **Feature Pyramid Network (FPN) enhances multi-scale feature detection**, allowing the system to accurately identify damages across different resolutions. The **Region Proposal Network (RPN)** generates candidate bounding boxes, which are refined and classified by the **ROI Head** to produce precise damage localization and segmentation masks.
- Post-detection, the system calculates the repair cost by analyzing the detected damage in relation to the vehicle’s value and age, considering carfully choosen damage factors. The results are visualized with detailed breakdowns, providing users with actionable insights.
  
- **Vehicle Object Detection:**  An additional car object detection pipeline with pretrained **Faster R-CNN** weights is employed to identify and isolate the vehicle in the image, ensuring that the above damage detection is focused on the correct region.

### 4. LLM ChatBot:
- **RAG-Based Information Retrieval:** The CarBot component employs a Retrieval-Augmented Generation (RAG) pipeline to handle user queries about their vehicles by processing car manuals or related web content. It uses **LangChain for document processing**, where car manuals are parsed and segmented using the Recursive Character TextSplitter. Web content is similarly processed through an UnstructuredURLLoader. The segmented text is embedded using OpenAI or Ollama embeddings, depending on the model selected, and **indexed with FAISS** (Facebook AI Similarity Search) to facilitate efficient retrieval. The prompt is carfully designed to return an answer only if it is present in the provided document, otherwise it lets the user know that the answer is not present in the uploaded documents.
  
**Query Processing:** When a user submits a query, the system first **rephrases it within the context of the chat history** using LLama 3.1/ GPT 4-o-mini. The rephrased question is then matched against the indexed content, and relevant segments are retrieved. These segments are fed into a language model to generate a precise, context-aware answer.

<div align="center">
   <img src="https://github.com/user-attachments/assets/76985443-ee97-4b38-a247-7dc15d3fdf73" alt="RAG pipeline" width="75%">
</div>

### 5. Database and File Management:
**SQLite Database:** User data, including account details, session information, and inference results, are stored in an SQLite database. The database is managed using SQLAlchemy, providing a robust and scalable solution for data persistence.

**Static Files:** Images, processed documents, and inference results are managed in structured directories, ensuring easy access and retrieval.

### 6. Error Handling and Logging:
**Logging:** The system employs detailed logging at various stages, ensuring that errors can be traced and addressed efficiently.

**Error Management:** The system is designed to handle common errors gracefully, providing users with clear feedback and suggestions for resolving issues.

## Installation

To set up the project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/Divyeshpratap/A.I.-AutoInspector.git
2. Navigate to the project directory and run the setup script to install necessary dependencies.
   ```bash
   cd A.I.-AutoInspector
   chmod +x setup.sh
   ./setup.sh

3. Set up your environment variable (.env file):
   ```bash
   FLASK_SECRET_KEY='your_flask_secret_key'
   GoogleMaps_API_KEY='your_google_maps_api_key'
   GoogleSearch_API_KEY='your_google_search_api_key'
   GoogleSearch_engine_id='your_google_search_engine_id'
   OPENAI_API_KEY='your_openai_api_key'


4. Execute the script using the following command:
   ```bash
   python create_admin.py [--username <admin_username>] [--email <admin_email>] [--password <admin_password>]
   python app.py

## Directory
### (1)Root:
```
${ROOT}/
│
├── carDDModel/                 # Damage inferencing model and associated weights
│   ├── dcn_plus_cfg_small.py    # Configuration file for the DCN model
│   └── checkpoint.pth           # Checkpoint weights for the DCN model
│
├── instance/                   # SQLite database
│   └── site.db
│
├── models/                     # Python modules for database, CV, NLP
│   ├── database.py             # Database models and setup
│   ├── cv/                     # Computer Vision models
│   │   └── cv_model.py         # CV model implementations
│   │
│   ├── nlp/                    # Natural Language Processing models
│   │   └── nlp_model.py        # NLP model implementations
│   │
│   └── __pycache__/            # Python cache files (excluded from README)
│
├── static/                     # Static assets for the web app
│   ├── css/                    # CSS files
│   │   └── style.css
│   ├── images/                 # Image files
│   │   ├── inferenceSample.png
│   │   ├── logo.png
│   │   ├── RAG_Flow.png
│   │   └── visualDamagePipeline.png
│   ├── js/                     # JavaScript files
│   │   └── script.js
│   └── results/                # Result images or data
│
├── templates/                  # HTML templates for Flask
│   ├── add_agent.html
│   ├── admin_dashboard.html
│   ├── admin_login_dontknow.html
│   ├── agent_dashboard.html
│   ├── agent_profile.html
│   ├── base.html
│   ├── chatbot_index.html
│   ├── detailed_analysis.html
│   ├── edit_user.html
│   ├── index.html
│   ├── login.html
│   ├── order_details.html
│   ├── register.html
│   ├── upload.html
│   ├── user_dashboard.html
│   ├── user_order_details.html
│   └── user_profile.html
│
├── uploads/                    # Uploaded files for chatbot and other features
│
├── utils/                      # Utility modules and scripts
│   ├── context_processor.py
│   ├── data_helpers.py
│   ├── filters.py
│   ├── geolocation.py
│   ├── helpers.py
│   ├── sqlalchemy_events.py
│   └── web_scraper.py
│
├── app.py                      # Main Flask application script
├── create_admin.py             # Script to create admin account
├── LICENSE                     # Project license
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── setup.sh                    # Script to set up the environment

```
## Data

The checkpoint file (pretrained weight) are placed in carDDModel folder.

## Result

### Image Pipeline

Below are the training result of the damage detection and segmentation model.
#### Training Metrics and Curve
For an object detection and segmentation model, several metrics help to understand the performance and behavior:
- **Loss RPN CLoss (Region Proposal Network Classification Loss):** Measures how well the Region Proposal Network (RPN) classifies anchor boxes as containing an object or background. It's critical for the model's ability to propose potential object locations.
- **Loss RPN BBox (Region Proposal Network Bounding Box Loss):** Measures the accuracy of the bounding box adjustments made by the RPN to refine anchor boxes into object proposals.
- **Loss Cls (Classification Loss):**  Evaluates the classification performance of the model in assigning detected objects to the correct categories.
- **Loss BBox (Bounding Box Loss):** Quantifies the error in the predicted bounding boxes relative to the ground truth boxes, affecting the accuracy of object localization.
- **Loss Mask (Mask Loss):** Measures the quality of the segmentation masks generated by the model, which outlines the detected objects.

<div align="center">
   <img src="https://github.com/user-attachments/assets/b33c7d7e-8014-412c-bf4a-cdb481b3a9e6" alt="Training Loss Metrics" width="75%">
</div>


Overall, as seen in the above image, the training loss across all components consistently decreases over the epochs and flattens towards the end. The total loss decreases from 0.851 at epoch 1 to 0.281 at epoch 2, and finally to 0.189 at epoch 31, showing a significant reduction, which reflects effective learning.
#### Validation Metrics and Curve:
**Mean Average Precision (mAP)** metrics are used for evaluating the performance of object detection models, especially on validation data. mAP measures how well the model detects and localizes objects in images by calculating the average precision at various Intersection over Union (IoU) thresholds.

<div align="center">
   <img src="https://github.com/user-attachments/assets/b638ba82-e987-4001-9675-bf86c8bbd315" alt="Validation Metrics" width="75%">
</div>

- **BBox mAP Small, Medium, and Large (Bounding Box mAP for Different Object Sizes):** These metrics measure the mAP separately for small, medium, and large objects, providing insight into how the model performs across various object scales. In the above graph, it can be seen that the mAP is higher for larger objects compared to smaller ones, which is a common trend in object detection models. The mAP graph shows generally improving trends for all three object sizes.
- **Segmentation mAP (Segmentation Mean Average Precision):** Segmentation mAP evaluates the model’s performance in segmenting objects, measuring how well the predicted masks match the ground truth masks. Segmentation mAP also shows a positive trend, reflecting improved segmentation quality.

The consistent decrease in training losses and the improving mAP values indicate that the model is effectively learning to detect and segment objects. The performance is especially strong for medium and large objects, while small object detection remains a typical challenge.

**Text Pipeline**

For Retrieval-Augmented Generation (RAG) based LLM chatbot models, **qualitative measures** are usually more indicative of the model's performance than quantitative metrics. Although the evaluation of the chatbot is still in progress, the quality of results improved significantly after the integration of history, allowing for refined question prompts by considering prior conversation context.

## Acknowledgment

This project uses resources and data from:
carDD dataset https://github.com/CarDD-USTC/CarDD-USTC.github.io

## Contact
For more information, contact [Divyesh Pratap Singh](https://www.linkedin.com/in/divyesh-pratap-singh/)

## License

This project is licensed under the [MIT License](LICENSE).
