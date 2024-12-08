import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image


# Load the dataset
data = pd.read_csv('Danh_gia_cleaned_updated.csv')
data['content_new'] = data['content_new'].fillna('')
data['sentiment'] = np.where(data['so_sao'] <= 3, 0, 1)  # 0 = Negative, 1 = Positive


# Features and targetstrea
X = data['content_new']
y = data['sentiment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

# Vectorizer for text data
count = TfidfVectorizer(max_features=5000)
X_train = count.fit_transform(X_train)
X_test = count.transform(X_test)

# Train the model
svc_model = SVC(C=10, gamma='scale', kernel='rbf', probability=True)
svc_model.fit(X_train, y_train)

# Predictions
y_predict = svc_model.predict(X_test)

# Evaluation
score_train = svc_model.score(X_train, y_train)
score_test = svc_model.score(X_test, y_test)
acc = accuracy_score(y_test, y_predict)
cm = confusion_matrix(y_test, y_predict, labels=[0, 1])
cr = classification_report(y_test, y_predict)
y_prob = svc_model.predict_proba(X_test)
roc = roc_auc_score(y_test, y_prob[:, 1])



# Save the model and vectorizer
pkl_filename = "svc_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(svc_model, file)

pkl_count = "count_vectorizer.pkl"
with open(pkl_count, 'wb') as file:
    pickle.dump(count, file)

df2 = pd.read_csv('San_pham.csv')
data['label'] = data['so_sao'].apply(lambda x: 'positive' if x >= 3 else 'negative')

def analyze_product_reviews(product_code):
    # Lọc các nhận xét liên quan đến sản phẩm
    product_reviews = data[data['ma_san_pham'] == product_code]

    # Lọc các nhận xét tích cực và tiêu cực
    positive_reviews = product_reviews[product_reviews['label'] == 'positive']
    negative_reviews = product_reviews[product_reviews['label'] == 'negative']

    # Hiển thị số lượng nhận xét tích cực và tiêu cực
    st.write(f"Số nhận xét tích cực: {len(positive_reviews)}")
    st.write(f"Số nhận xét tiêu cực: {len(negative_reviews)}")

    # Tạo WordCloud cho nhận xét tích cực
    positive_reviews['content_new'] = positive_reviews['content_new'].fillna('').astype(str)
    negative_reviews['content_new'] = negative_reviews['content_new'].fillna('').astype(str)

    positive_text = ' '.join(positive_reviews['content_new'])
    positive_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(positive_text)

    negative_text = ' '.join(negative_reviews['content_new'])
    negative_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_text)

    # Hiển thị WordCloud
    st.write("### WordCloud")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positive Sentiment")
        st.image(positive_wc.to_array())
    with col2:
        st.subheader("Negative Sentiment")
        st.image(negative_wc.to_array())

    # Tạo mô hình TF-IDF
    product_reviews['content_new'] = product_reviews['content_new'].fillna('')
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    tfidf_matrix = vectorizer.fit_transform(product_reviews['content_new'])

    # Lấy các từ khóa quan trọng
    features = vectorizer.get_feature_names_out()
    st.write("### Các từ khóa chính:")
    st.write(features)

    # Trực quan hóa số lượng nhận xét theo ngày
    product_reviews['ngay_binh_luan'] = pd.to_datetime(product_reviews['ngay_binh_luan'])
    product_reviews['day'] = product_reviews['ngay_binh_luan'].dt.date
    daily_reviews = product_reviews.groupby('day').size()

    st.write("### Số lượng nhận xét theo ngày")
    st.line_chart(daily_reviews)

    # Trực quan hóa số sao trung bình theo ngày
    average_rating = product_reviews.groupby('day')['so_sao'].mean()

    st.write("### Sao trung bình theo ngày")
    st.line_chart(average_rating)

    # Hiển thị thông tin sản phẩm
    product_info = df2[df2['ma_san_pham'] == product_code].iloc[0]
    st.write("### Thông tin sản phẩm")
    st.write(f"Tên sản phẩm: {product_info['ten_san_pham']}")
    st.write(f"Giá bán: {product_info['gia_ban']}")
    st.write(f"Giá gốc: {product_info['gia_goc']}")
    st.write(f"Phân loại: {product_info['phan_loai']}")
    st.write(f"Điểm trung bình: {product_info['diem_trung_binh']}")


# Streamlit Interface
st.title("Text Classification: Positive vs Negative Sentiment")

# Menu for navigation
menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Team members: Cao Anh Hào - Phan Văn Minh""")
st.sidebar.write("""#### Instructor: Khuất Thùy Phương """)
st.sidebar.write("""#### Date: 8/12/2024""")

if choice == 'Business Objective':
    # Submenu for 'Business Objective' to switch between tabs
    objective_choice = st.radio("Sections:", options=["Tổng quan", "Phân tích sản phẩm", "Algorithm"])

    if objective_choice == "Tổng quan":
        st.subheader("Sơ lược về Hasaki")
        # Display banner image
        banner = Image.open('hasaki_banner_2.jpg')
        st.image(banner, caption="Sơ lược về Hasaki", use_container_width=True)

        st.write(""" 
        Hasaki là một chuỗi cửa hàng phân phối mỹ phẩm đang phát triển mạnh mẽ tại Việt Nam, với mạng lưới cửa hàng ngày càng mở rộng trên toàn quốc. Ngoài việc tập trung vào việc phát triển hệ thống bán lẻ truyền thống, Hasaki còn chú trọng vào việc nâng cấp và cải thiện trải nghiệm mua sắm trực tuyến trên trang web hasaki.vn. Mỗi ngày, Hasaki tiếp nhận hàng ngàn lượt đánh giá và phản hồi từ khách hàng về các sản phẩm của mình.
        """)
        st.subheader("Quản lý và Phân tích Ý Kiến Khách Hàng: Cải Thiện Sản Phẩm và Xây Dựng Thương Hiệu")
        # Display second image about product management
        product_management_image = Image.open('trang1.png')
        st.image(product_management_image, caption="Quản lý sản phẩm thông qua đánh giá ý kiến khách hàng", use_container_width=True)
        
        st.write("""
        Việc quản lý và thu thập ý kiến đánh giá từ khách hàng là yếu tố then chốt trong việc cải thiện sản phẩm, mở rộng tệp khách hàng và xây dựng thương hiệu. Nhờ vào hệ thống trực tuyến, Hasaki có thể thu thập ý kiến một cách khách quan từ người dùng thực tế. Những dữ liệu giá trị này giúp doanh nghiệp phân tích sâu hơn, từ đó phát triển hệ thống dự đoán tâm lý khách hàng trong tương lai một cách chính xác và hiệu quả hơn.
        """)
        
        # Display third image related to sentiment analysis
        sentiment_image = Image.open('danhgia.png')
        st.image(sentiment_image, caption="Sentiment Analysis", use_container_width=True)
        st.subheader("Mô Hình Dự Đoán Phản Hồi Khách Hàng: Phân Tích Cảm Xúc và Tác Động đến Sản Phẩm & Dịch Vụ")
        st.write("""
        Phân tích cảm xúc (Sentiment Analysis) là một kỹ thuật trong xử lý ngôn ngữ tự nhiên giúp xác định cảm xúc trong các văn bản, ví dụ như đánh giá sản phẩm, nhận xét của khách hàng. Với việc ứng dụng Sentiment Analysis, Hasaki có thể nhanh chóng phân loại và phân tích các đánh giá từ khách hàng thành các nhóm cảm xúc tích cực hoặc tiêu cực. Điều này giúp doanh nghiệp nhận diện kịp thời các vấn đề hoặc cải tiến sản phẩm, từ đó nâng cao sự hài lòng của khách hàng và cải thiện trải nghiệm mua sắm.
        """)
        st.write("""
       Mục tiêu của mô hình dự đoán này là giúp Hasaki.vn và các đối tác nhận diện nhanh chóng các phản hồi của khách hàng về sản phẩm và dịch vụ (tích cực, tiêu cực). Điều này sẽ hỗ trợ họ trong việc cải thiện chất lượng sản phẩm và dịch vụ, từ đó nâng cao sự hài lòng và trải nghiệm của khách hàng
        """)

    elif objective_choice == "Phân tích sản phẩm":
        st.subheader("Phân tích sản phẩm")
        # Input field for entering product code
        product_code_input = st.text_input("Nhập mã sản phẩm:")
        if product_code_input:
            try:
                product_code_input = int(product_code_input)  # Ensure the input is an integer
                analyze_product_reviews(product_code_input)
            except ValueError:
                st.error("Hãy nhập đúng mã sản phẩm.")
        
        # Get the list of available product codes for the selectbox
        product_codes = data['ma_san_pham'].unique()
        
        # Create a selectbox for the user to choose a product code
        product_code_select = st.selectbox("Hoặc chọn mã sản phẩm từ danh sách mã sản phẩm:", product_codes)

        # Call the analyze_product_reviews function when a product code is selected
        if product_code_select is not None:
            analyze_product_reviews(product_code_select)

    elif objective_choice == "Algorithm":
        st.subheader("Thuật toán")
        st.write("Thông tin về thuật toán sẽ được cung cấp sau.")

elif choice == 'Build Project':
    # Add Data Overview
    st.write("##### Data Overview")

    # Show first 3 rows of relevant columns (content and rating)
    st.subheader("First 3 rows of content and rating")
    st.write(data[['noi_dung_binh_luan', 'so_sao']].head(3))

    # Show a summary of the target variable 'so_sao'
    st.subheader("Summary of the 'so_sao' ratings")
    st.write(data['so_sao'].value_counts())

    # Display the processed data column 'content_new'
    st.subheader("Processed Data: 'content_new'")
    st.write(data['content_new'].head(3))  # Display first 3 rows of 'content_new'

    # Optional: Display the overall data summary (statistics, missing data, etc.)
    st.subheader("Overall Data Summary")
    st.write(data.describe())  # Show basic statistics of numeric columns
    st.write(data.info())  # Show info for column types and missing data
    
    # You can add more visualizations as needed, such as histograms for 'so_sao'
    st.write("##### Distribution of Ratings (so_sao)")

    # Create a figure and axis to avoid the deprecation warning
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data['so_sao'], kde=True, ax=ax)  # Pass the ax to the plot to ensure it uses the correct axis
    st.pyplot(fig)  # Pass the figure object to st.pyplot
    
    # Model Evaluation Section
    st.write("##### Model Evaluation")
    st.write(f"##### Training score: {round(score_train, 2)} vs Test score: {round(score_test, 2)}")
    st.write(f"##### Accuracy: {round(acc, 2)}")

    # Display Confusion Matrix
    st.write("###### Confusion Matrix:")
    st.code(cm)

    # Plot confusion matrix using seaborn heatmap
    cm_df = pd.DataFrame(cm, index=['Negative', 'Positive'], columns=['Negative', 'Positive'])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16}, linewidths=1, linecolor='black')
    plt.title("Confusion Matrix", fontsize=16)
    st.pyplot(plt)

    # Display Classification Report
    st.write("###### Classification report:")
    st.code(cr)

    # Display ROC AUC Score
    st.write(f"###### ROC AUC score: {round(roc, 2)}")

    # Plot ROC curve
    st.write("###### ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.plot(fpr, tpr, marker='.')
    st.pyplot(fig)

elif choice == 'New Prediction':
    st.subheader("Make a Prediction")
    
    # Flag to check if the content is ready for prediction
    flag = False
    lines = None
    
    # Radio button to choose between upload and input
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    
    # File upload option
    if type == "Upload":
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)  # Display the data in the uploaded file
            lines = lines[0]  # Get the first column for prediction
            flag = True
    
    # Manual text input option
    if type == "Input":
        content = st.text_area(label="Input your content:")
        if content != "":
            lines = np.array([content])
            flag = True
    
    # If content is available for prediction
    if flag:
        # Display the content that will be predicted
        st.write("Content:")
        if len(lines) > 0:
            st.code(lines)  # Display the content in code block
            
            # Perform prediction
            x_new = count.transform(lines)  # Transform the input text
            y_pred_new = svc_model.predict(x_new)  # Predict using the model
            
            # Determine sentiment based on prediction result
            sentiment = "Positive" if y_pred_new == 1 else "Negative"
            
            # Display the prediction result
            st.write(f"Prediction Result: {sentiment}")
            
            # Show a dataframe with prediction results for better clarity
            prediction_df = pd.DataFrame({
                'Content': lines,
                'Prediction': [sentiment]
            })
            st.dataframe(prediction_df)  # Display the prediction result in table format
