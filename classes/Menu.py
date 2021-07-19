# Created by Antonio Di Mariano (antonio.dimariano@gmail.com) at 17/12/2019

from classes.PredictionModels import RandomForest, XGboost
from utils.build_submision_csv import build_csv
from classes.Driver import AirbnbData
DATA_ANALYSIS_GRAPH = 'data_analysis_graphs/'
import datetime
import calendar


class Questions:

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.driver = AirbnbData()
        self.questions_and_answers = {
            0: {
                "question": "\n\nWhat do you want to do ?\n"
                            "(a) Investigate data and describe data.\n"
                            "(b) Investigate data and plot graphs.\n"
                            "(c) Investigate data, plot data, run RandomForest Model and generate cvs\n"
                            "(d) Investigate data, plot data, run XGBoost Model and generate cvs (it might take time) .\n"
                            "(e) Keep my money and give me everything! ( it will take time!).’\n\n"
            }

        }

    def ask_question(self, question_id=0):
        """
        This functions asks the question selected by the given question_id
        If the answer is not valid, the same question will be asked again until a valid answer will be provvided.

        :param question_id:
        :return:
        """
        allowed_answers = ['a', 'b', 'c', 'd','e']

        action_to_take = {
            'a': ['investigate_all_data', 'describe_data'],
            'b': ['investigate_all_data', 'plot_all_data'],
            'c': ['investigate_all_data', 'plot_all_data', 'train_model', 'randomforest_and_build_csv'],
            'd': ['investigate_all_data', 'plot_all_data', 'train_model', 'xgboost_and_build_csv'],
            'e': ['investigate_all_data', 'plot_all_data', 'train_model', 'randomforest_and_build_csv',
                  'xgboost_and_build_csv']
        }

        answer = input(self.questions_and_answers[question_id].get('question'))
        if answer in allowed_answers:
            print("Your answer:", answer)
            function_to_run = action_to_take[answer]
            for function in function_to_run:
                getattr(self, function)()

        else:
            print("Please type a,b,c or d\n")
            self.ask_question(question_id=0)


    def describe_data(self):
        print("[] Describing data")
        self.driver.describe_data()
        print("[*] Data describing completed")
    def investigate_all_data(self):
        print("[] Investigating the Dataset")
        self.driver.investigate_the_time_users_spend_before_make_a_reservation()
        self.driver.investigate_most_active_signup_and_bookings_years_month()
        self.driver.investigate_users_age()
        print("[*] The Dataset has been cleaned. We are good to go!")

    def plot_all_data(self):
        print("[] I am plotting..please wait")
        self.driver.plot_data_about_age_and_gender()
        self.driver.plot_data_about_time_spent_on_the_website()
        self.driver.plot_data_about_booking_period()
        self.driver.plot_distribution_of_categorical_data()
        print("[*] Your graphs are ready in the folder ",DATA_ANALYSIS_GRAPH)

    def train_model(self):
        new_df = self.driver.dummify_categorical_features()
        self.driver.investigate_gender(dataframe=new_df)
        self.x_train, self.y_train, self.x_test = self.driver.training_model(new_df=new_df)

    def randomforest_and_build_csv(self):

        now = datetime.datetime.utcnow()
        timestamp = calendar.timegm(now.utctimetuple())
        build_csv(test_users=self.driver.test_users, Y_pred=RandomForest().predict(x_train=self.x_train,
                                                                                   y_train=self.y_train,
                                                                                   x_test=self.x_test),
                  filename='Airbnb_prediction_randomforest_'+str(timestamp)+".csv")

    def xgboost_and_build_csv(self):
        now = datetime.datetime.utcnow()
        timestamp = calendar.timegm(now.utctimetuple())
        build_csv(test_users=self.driver.test_users, Y_pred=XGboost().predict(x_train=self.x_train,
                                                                              y_train=self.y_train,
                                                                              x_test=self.x_test),
                  filename='Airbnb_prediction_xgboost_'+str(timestamp)+".csv")
