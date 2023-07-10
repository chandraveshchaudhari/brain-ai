from execution import SentimentDataExecutor


class TweetPredictivePower(SentimentDataExecutor):
    def __init__(self, tweets, stock_price_direction):
        super().__init__()
        self.stock_price_direction = stock_price_direction
        self.tweets = tweets
        self.data = []

    def collecting_results_into_applicable_format(self):
        # get applicable format from prediction-techniques-comparison
        pass

    def execute(self):
        """[{'DateTime':"", 'tweet':"", 'direction':""}]

        """
        result = []
        for tweet in self.tweets:
            result = self.average_sentiment_score()

        return result

