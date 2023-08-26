async function logMovies(data) {
    const url = "http://127.0.0.1:5000/";
    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ "user_input": data })
    });

    const analyze_sentiment = await response.json();
    const sentiment = analyze_sentiment.sentiment;
    
    console.log("Sentiment:", sentiment);
}