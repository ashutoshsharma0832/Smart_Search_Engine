import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from courses import load_course_data

#environment variables
load_dotenv()
hugging_face_api_token = os.getenv("HUGGING_FACE_API_TOKEN")

#sentence transformer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
sentence_transformer = SentenceTransformer(model_name)

def search_courses(user_query, course_data, top_k=5):
    # Prepare course information for embedding
    course_info = [
        f"{course['title']} - {course['description']} - Skills: {', '.join(course['skills'])}"
        for course in course_data
    ]

    # Get embeddings
    query_embedding = sentence_transformer.encode([user_query])
    course_embeddings = sentence_transformer.encode(course_info)

    # Calculate similarities
    similarities = cosine_similarity(query_embedding, course_embeddings)[0]

    # Get top k courses
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        course = course_data[idx]
        similarity_score = similarities[idx]
        results.append({
            **course,
            "similarity": similarity_score
        })

    return results

def main():
    st.set_page_config(
        page_title="Analytics Vidhya Course Search",
        page_icon="📚",
        layout="wide"
    )

    # Header
    st.title("📚 Analytics Vidhya Smart Course Search")
    st.write("Find the perfect free course for your learning journey!")

    # Sidebar filters
    st.sidebar.title("Filter Options")

    # Load course data
    course_data = load_course_data()

    # Get unique categories
    categories = sorted(list(set(cat for course in course_data for cat in course['category'])))
    selected_categories = st.sidebar.multiselect(
        "Filter by Category",
        categories
    )

    # Get unique skills
    all_skills = sorted(list(set(
        skill for course in course_data
        for skill in course['skills']
    )))
    selected_skills = st.sidebar.multiselect(
        "Filter by Skills",
        all_skills
    )

    # Filter courses based on selections
    filtered_courses = course_data
    if selected_categories:
        filtered_courses = [
            course for course in filtered_courses
            if any(cat in selected_categories for cat in course['category'])
        ]
    if selected_skills:
        filtered_courses = [
            course for course in filtered_courses
            if any(skill in selected_skills for skill in course['skills'])
        ]

    # Search input
    user_query = st.text_area(
        "Describe what you want to learn:",
        placeholder="E.g., I want to learn Python for data science with focus on machine learning and deep learning...",
        height=100
    )

    # Search button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        search_button = st.button("🔍 Search Courses")

    # Process search
    if search_button and user_query:
        if not filtered_courses:
            st.warning("No courses match your filter criteria. Try adjusting the filters.")
        else:
            with st.spinner("Searching for the best courses..."):
                results = search_courses(user_query, filtered_courses)

                st.subheader("📋 Recommended Courses")
                for i, result in enumerate(results, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"""
                            ### {i}. {result['title']}
                            **Categories:** {', '.join(result['category'])}  
                            **Relevance Score:** {result['similarity']:.2%}
                            {result['description']}
                            **Skills you'll learn:** {', '.join(result['skills'])}
                            [View Course]({result['url']})
                            ---
                            """)

    # Display help information at the bottom
    with st.expander("ℹ️ How to use this search"):
        st.markdown("""
        1. Use the sidebar filters to narrow down courses by category or specific skills (optional)
        2. Enter a description of what you want to learn in the text box above
        3. Click the 'Search Courses' button
        4. Browse through the recommended courses
        5. Click on 'View Course' to access the course on Analytics Vidhya
        **Tips for better results:**
        - Be specific about what you want to learn
        - Include your experience level
        - Mention specific technologies or skills you're interested in
        - Use the sidebar filters to focus on specific categories or skills
        """)

if __name__ == "__main__":
    main()
