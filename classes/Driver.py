# Created by Antonio Di Mariano (antonio.dimariano@gmail.com) at 15/12/2019
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

country_num_dic = {'NDF': 0, 'US': 1, 'other': 2, 'FR': 3, 'IT': 4, 'GB': 5, 'ES': 6, 'CA': 7, 'DE': 8, 'NL': 9,
                   'AU': 10, 'PT': 11}

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_ANALYSIS_GRAPH = 'data_analysis_graphs/'
DATA_DIR = 'airbnb-recruiting-new-user-bookings'
age_file = DATA_DIR + "/age_gender_bkts.csv"
countries_file = DATA_DIR + "/countries.csv"
sessions_file = DATA_DIR + "/sessions.csv"
train_users = DATA_DIR + "/train_users_2.csv"
test_users = DATA_DIR + "/test_users.csv"


class AirbnbData:
    def __init__(self):
        self.train_users = None
        self.test_users = None
        self.age = None
        self.countries = None
        self.sessions = None
        self.label_enc = LabelEncoder()
        self.joined_data = None
        self.load_data()

    def load_data(self):
        """
        Load all dataset
        """
        self.train_users = self._load_data_from_csv(csv_file_full_path=train_users)
        self.test_users = self._load_data_from_csv(csv_file_full_path=test_users)
        self.age = self._load_data_from_csv(csv_file_full_path=age_file)
        self.sessions = self._load_data_from_csv(csv_file_full_path=sessions_file)
        self.countries = self._load_data_from_csv(csv_file_full_path=countries_file)

        # Here I join the test and train data to work with them at the same time
        self.joined_data = self.train_users.append(self.test_users, ignore_index=True, sort=True)

        print("[*] Initial AirBnB Dataset Loaded")

    def _load_data_from_csv(self, csv_file_full_path):
        """
        This method loads data from the given cvs file
        """
        try:
            data = pd.read_csv(csv_file_full_path)

            return data
        except Exception as error:
            print("Error loading csv file:", error)

    def describe_data(self):
        print("TRAIN DATA DESCRIBE")
        print(self.train_users.describe())
        print("TRAIN DATA INFO")
        print(self.train_users.info())

        print("TEST DATA DESCRIBE")
        print(self.test_users.describe())
        print("TEST DATA INFO")
        print(self.test_users.info())
        self.extract_quantitative_and_qualitative_variables(data_set=self.train_users, name='TRAIN')
        self.extract_quantitative_and_qualitative_variables(data_set=self.test_users, name='TEST')

    def extract_quantitative_and_qualitative_variables(self, data_set, name, drop_id=True):
        """
        This method creates two list. one for the numerical values and one for the categorical values of the
        given data_set
        :param data_set:
        :param drop_id:
        :return:
        """
        # Categorization. divide the data into numerical ("quan") and categorical ("qual") features
        quantitative = list(data_set.loc[:, data_set.dtypes != 'object'].columns.values)
        qualitative = list(data_set.loc[:, data_set.dtypes == 'object'].columns.values)
        print(name, " quantitative variables:", quantitative)
        print(name, " qualitative variables:", qualitative)

    def investigate_the_time_users_spend_before_make_a_reservation(self):
        """
        This method creates 3 new columns so as to investigate the time a user spent before made a reservation

        """
        self.joined_data['timestamp_first_active'] = pd.to_datetime(
            (self.joined_data.timestamp_first_active // 1000000), format='%Y%m%d')
        self.joined_data['date_first_booking'] = pd.to_datetime(self.joined_data['date_first_booking'])
        self.joined_data['time_to_booking'] = self.joined_data['date_first_booking'] - self.joined_data[
            'timestamp_first_active']

    def investigate_most_active_signup_and_bookings_years_month(self):
        """
        This method checks every month and year of the follow columns
            - date_first_booking
            - date_account_created
            - timestamp_first_active

        in order to spot in what period of the year  most of the reservations  happened.
        It creates the following new columns so as to hold those values

            - month_booking
            - year_booking
            - date_account_created
            - month_account_created
            - year_account_created
            - month_first_time_active
            - year_first_time_active

        """
        self.joined_data['month_booking'] = self.joined_data.date_first_booking.dt.month
        self.joined_data['year_booking'] = self.joined_data.date_first_booking.dt.year
        self.joined_data['date_account_created'] = pd.to_datetime(self.joined_data['date_account_created'])
        self.joined_data['month_account_created'] = self.joined_data.date_account_created.dt.month
        self.joined_data['year_account_created'] = self.joined_data.date_account_created.dt.year
        self.joined_data['month_first_time_active'] = self.joined_data.timestamp_first_active.dt.month
        self.joined_data['year_first_time_active'] = self.joined_data.timestamp_first_active.dt.year

    def investigate_users_age(self):
        """
        From the analysis of the age columns has emerged a lot of no-realistic values, such as age > 100 years old.
        So in order to clean the data I set any age value > 90 and < of 18 to NaN.
        I am not saying a 90 y.o guy cannot travel, but for the sake of this exercise I am ruling those values out.
        Also, I replace the missing data with the median value. I know there are some other methods but from what
        I've learned this method seems to be one of the most common
        """
        try:

            self.joined_data.age[self.joined_data.age > 90] = np.nan
            self.joined_data.age[self.joined_data.age < 18] = np.nan
            print("Median AGE:",self.joined_data.median())
            # Replace Missing age data with the mean
            self.joined_data.loc[self.joined_data['age'].isnull(), 'age'] = self.joined_data.age.median()

        except Exception as error:
            print("Exception investigate_users_age:", error)
            return 0
    def dummify_categorical_features(self):

        """
        So many features in the data set are categorical and they can be dummify so the model can do its job


        """
        try:
            df1 = self.joined_data.drop(
                ['date_first_booking', 'time_to_booking', 'month_booking', 'year_booking', 'date_account_created',
                 'timestamp_first_active', 'timestamp_first_active', 'country_destination', 'id'], axis=1)
            ndf = pd.get_dummies(df1, columns=['affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
                                               'first_browser', 'first_device_type', 'language', 'signup_app',
                                               'year_first_time_active'
                , 'signup_flow', 'signup_method', 'month_account_created', 'year_account_created',
                                               'month_first_time_active'],
                                 drop_first=True, dtype='float16')
            return ndf

        except Exception as error:
            print("Exception dummify_categorical_features:", error)
            return 0

    def investigate_gender(self, dataframe):
        """
        The gender column has categorical values and a lot of missing values too.
        This methods deals with the gender columns so to have those values in numeric form.
        I replace the missing value with -1

        """
        try:
            dataframe.loc[dataframe.gender == '-unknown-', 'gender'] = -1
            dataframe.loc[dataframe.gender.isnull(), 'gender'] = -1
            dataframe.loc[dataframe.gender == '-unknown-', 'gender'] = -1
            dataframe.loc[dataframe.gender.isnull(), 'gender'] = -1

            gender_enc = {'FEMALE': 0,
                          'MALE': 1,
                          'OTHER': 2,
                          '-unknown-': -1,
                          -1: -1,
                          1: 1,
                          0: 0,
                          2: 2

                          }
            for data in [dataframe]:
                data.gender = data.gender.apply(lambda x: gender_enc[x])

        except Exception as error:
            print("Exception investigate_gender:", error)
            return 0

    def training_model(self, new_df):
        try:
            """
            This methods builds the required training and testing sets
    
            The function dataframe.columns.difference()
    
            gives you complement of the values that you provide as argument.
            It can be used to create a new dataframe from an existing dataframe with exclusion of some columns
            So df_with_no_country_destination has all the columns except the country_destination.
    
            """
            df_with_no_country_destination = new_df[new_df.columns.difference(['country_destination'])]

            """
            reset_index() is a method to reset index of a Data Frame. 
            Since the country_destination column is categorical I sue the reset_index() to change the column's values back 
            to integer indices.
            """

            df_with_country_destination_integer_indices = self.joined_data['country_destination'].reset_index()

            """
            Here I split the dataframe in two pieces, dropping the index from the y_train
            """
            x_train = df_with_no_country_destination.iloc[0:213451, :].values
            y_train = df_with_country_destination_integer_indices.iloc[0:213451, :].drop('index', axis=1).values
            x_test = df_with_no_country_destination.iloc[213451:, :].values

            return x_train, y_train, x_test
        except Exception as error:
            print("ERROR in training_model:", error)

    def plot_distribution_of_categorical_data(self):
        """
        This method plot a graph using all the categorical variables related to users behaviour
        """
        try:
            categorical_variables = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider'
                , 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'country_destination']

            for x in categorical_variables:
                sns.countplot(x=x, data=self.joined_data, palette='RdBu')
                plt.ylabel('Users Qty')
                plt.title('Users ' + x + ' Distribution')
                plt.xticks(rotation='vertical')
                plt.savefig(DATA_ANALYSIS_GRAPH + 'plot' + str(x) + '.png')

        except Exception as error:
            print("ERROR in plot_distribution_of_categorical_data:", error)

    def plot_data_about_booking_period(self):
        """
        This method gets the columns created in the ... function and plot these  values

        self.joined_data['month_booking'] = self.joined_data.date_first_booking.dt.month
        self.joined_data['year_booking'] = self.joined_data.date_first_booking.dt.year
        self.joined_data['month_account_created'] = self.joined_data.date_account_created.dt.month
        self.joined_data['year_account_created'] = self.joined_data.date_account_created.dt.year
        """
        try:
            for x in ['month_booking', 'year_booking', 'month_account_created', 'year_account_created']:
                sns.countplot(x=x, data=self.joined_data)
                plt.xticks(rotation='vertical')
                plt.savefig(DATA_ANALYSIS_GRAPH + 'plot' + str(x) + '.png')
        except Exception as error:
            print("ERROR in plot_data_about_booking_period:", error)

    def plot_data_about_time_spent_on_the_website(self):
        """

        This method plots the time users spent on the website
        """
        try:
            sessions2 = self.sessions.groupby('user_id').sum()
            df2 = self.joined_data.merge(sessions2, left_on='id', right_on='user_id', how='inner')

            secs = []
            counts = []
            for x in df2.country_destination.unique():
                dfndf = df2[df2.country_destination == x]
                dfndf['hour'] = dfndf.secs_elapsed // 3600
                counts.append(dfndf.id.count())
                secs.append(dfndf.hour.mean())

            sns.set_context('notebook')
            sns.scatterplot(x=counts, y=secs, hue=df2.country_destination.unique())
            plt.xlabel('Users Qty')
            plt.ylabel('Mean Hours Users Spends on the Website')
            plt.title('Web Sessions Data of Users')
            plt.savefig(DATA_ANALYSIS_GRAPH + 'plot Web Sessions Data of Users.png')

        except Exception as error:
            print("ERROR in plot_data_about_time_spent_on_the_website:", error)

    def plot_data_about_age_and_gender(self):
        """
        This method plots a barplot graph where the booking data are grouped by Age and Gender
        """
        try:
            graph = self.age.groupby(['age_bucket', 'gender']).sum().reset_index().sort_values(
                'population_in_thousands')
            sns.barplot(x='age_bucket', y='population_in_thousands', data=graph)
            plt.xticks(rotation='vertical')
            plt.title('Booking Data grouped by Age and Gender')
            plt.savefig(DATA_ANALYSIS_GRAPH + 'age_and_gender_booking.png')
        except Exception as error:
            print("ERROR in plot_data_about_age_and_gender:", error)
