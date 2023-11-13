# LinearRegression---INT3103

## Introduction

A project help you to predict your salary with your informations (such as Working Experience, calculate by month or year).

## Technologies

**Project uses:**

+ Visual Studio Code (VSCode).
+ NumPy 1.26.0.
+ Matplotlib 3.8.0.
+ Scikit-Learn - A popular library for Machine Learning.

## Set up environment 
- **Linux**
    - Turn on your terminal.
    - Download python to your computer
        - Type:
        ```
            $ sudo apt-get update
            $ sudo apt-get install python3.6
        ```
    - Download numpy
        - Type:
        ```
            $ python3 install numpy
        ```
    
    - Download matplotlib
        ```
            $ python3 install matplotlib
        ```

    - Download sckit-learn library
        ```
            $ python3 install -U scikit-learn
        ```
    
## Compile and Run
- You should run main.py (the main code) with one of data (in directory data). For example, when you run with data "YearExperience_Salary_1.data":
    ```
        python3 main.py data/YearExperience_Salary_1.data
    ```
    
- It will return the best model for this data, then request you to predict. You shoud type information to get your result - the expected salary.

_ If you have new data, you can extract your csv data by file extractData.py:
    ```
        python3 extractData.py csv/yourDataName.csv > data/yourDataName.data
    ```