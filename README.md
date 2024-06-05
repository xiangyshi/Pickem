# [Pick'em!](https://xiangyshi.github.io/Pickem/)
## By Xiangyu(Leo) Shi

### Esports!
Although esports differs from traditional sports like soccer or basketball, the fans are no less enthusiastic while watching a game, hoping the team they root for will prevail in the end. Among the popular competitive games currently, the League of Legends Worlds Championship is undoubtedly the most anticipated esports event of the year.

Each year, different leagues (usually from different countries/continents) compete internally and then select the best few teams to attend the World Championship series. During the championship, fans from all over the world can attend an event called "Pick'em," which is a prediction game that allows fans to guess the team ranks during playoffs and the winners of each stage of the knockout.

While making predictions based solely on the teams and their performance may be challenging, we can take this down a notch and use the first 15 minutes of gameplay for each match during the championship to make predictions about which team will prevail!

I have gathered data for the League of Legends - Pro Series (provided by [Oracle's Elixir](oracleselixir.com)) from all of the major leagues for the past three years, and we will use several features from this dataset to predict the 2023 Worlds matches (which occurred last November - December).

### Mechanics of the Game

In a match of League of Legends, two teams (red/blue side), consisted of 5 players each, compete against each other by gaining experience, gold, completing objectives, and win by destroying the opponent's base. The game is generally broken down into 3 phases: \
- The laning phase (early game): this is usually referred to the first 15 minutes of the game, where each player strives to gain more advantage over their opponent by farming for gold, experience, and early objectives. At this time, the characters controlled by the players are generally weak, and less prone to have significant advantages over the other team.
- The pushing phase (mid game): this is referred to the middle section of the game, where most players have gathered enough resources to begin grouping for team fights, mid game objectives become increasingly important, such as turrut pushing (which protects the opponents bases), dragons (which provides positive effects for the team), and heralds (which gives the team a significant advantage while pushing a turret).
- The late game: most of the time, it is pretty clear at this stage which team would win the match, however, there are still uncertainties as one fatal mistake may lead to your team's defeat, as all the players are much stronger, and can push turrets efficiently.

In this project, we will only focus on the gameplay for the early game, and use data gathered during this period to predict the winning chance of a given team.

### Selection of features

As mentioned previously, we will only consider data from the first 15 minutes of any given match, so the following columns from the dataset is selected:

| Feature Name | type | Description |
|:------------|---|-----------:|
| `goldat10`  | `int` | Ally total gold at 10 min |
| `opp_goldat10` | `int` | Opponent total gold at 10 min |
| `xpat10`  | `float` |Ally total xp at 10 min |
| `opp_xpat10` | `float` | Opponent total xp at 10 min  |
| `csat10`  | `int` |Ally total minion kill at 10 min |
| `opp_csat10` | `int` | Opponent total minion kill at 10 min |
| `killsat10` | `int` | Ally total champion(opponent character) kill at 10 min |
| `opp_killsat10`| `int` | Opponent total champion(ally character) kill at 10 min |
| `goldat15`  | `int` |Ally total gold at 15 min |
| `opp_goldat15` | `int` | Opponent total gold at 15 min |
| `xpat15`  | `float` | Ally total xp at 15 min |
| `opp_xpat15`  | `float` | Opponent total xp at 15 min |
| `csat15`  | `int` | Ally total minion kill at 15 min |
| `opp_csat15`  | `int` | Opponent total minion kill at 15 min |
| `killsat15`  | `int` | Ally total champion(opponent character) kill at 15 min |
| `opp_killsat15` | `int` | Opponent total champion(ally character) kill at 15 min |
| `firstdragon` | `boolean`| `True` if first dragon was obtained by ally, else `False` |
| `result` | `boolean` | `True` if ally victory, else `False` |
| `teamname`* | `string` | Ally teamname |

\* only in final model
### Feature Engineering

The following features were crafted from existing columns: 

| Feature Name | type | Description |
|:------------|---|-----------:|
| `goldat10pct`  | `float` | Ally weighted gold advantage at 10 min|
| `xpat10pct` | `float` | Ally weighted xp advantage at 10 min |
| `csat10pct`  | `float` | Ally weighted cs advantage at 10 min |
| `goldat15pct` | `float` | Ally weighted gold advantage at 15 min  |
| `xpat15pct`  | `float` | Ally weighted xp advantage at 15 min |
| `csat10pct` | `float` | Ally weighted cs advantage at 15 min |

Each of the aforementioned features were calculated using the following formula:
Define x_0 = `(feature)at(time)`, x_1 = `opp_(feature)at(time)`, the weighed advantage is (x_0 - x_1) / x_0

Additionally, the `string` type feature, `teamname`, is one-hot-encoded (this is for the final model).

### Data Exploration and Visualization

#### Distribution of winrate with ally obtaining the first dragon objective
<iframe
  src="plots/dragon.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


#### Distribution of outcome with gold advantages
<iframe
  src="plots/gold10.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
<iframe
  src="plots/gold15.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### Distribution of outcome with weighted xp advantages
<iframe
  src="plots/xp10pct.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
<iframe
  src="plots/xp15pct.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

We can clearly see that there is some correlation between early game advantage and the outcome. However, this is not a direct relation, and as games progress, a single play made by a team can cause a huge difference.

### Simple Baseline
The distributions we have seen here can be fitted with a logistic curve, and thus we can use `LogisticRegression` from Scikit-Learn to define a baseline model. To do so, we need to seperate our data into training and testing sets, and also the observations for the Worlds 23 games for final scoring.

The source code is listed here for convenience:
```

# Define custom transformer for dropping columns
# Drop teams with less than 10 games
class Dropper():
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self 

# Define baseline model
baseline = Pipeline([
    ('ColumnDrop', Dropper(['side', 'gameid', 'position', 'league', 'teamname'])),
    ('TextEncoding', encoder),
    ('Model', LogisticRegression(max_iter=3000, C=0.1, solver='saga', random_state=42))
])

```

As a result, the baseline model is able to produce a score of ~0.75 for the testing set, and ~0.72 for the Worlds set. A fair prediction.

### Ensemble Model

For more in-depth prediction, I have selected the `RandomForestClassifier`. This ensemble model is less prone to overfitting, and fairly easy to use. To provide more accuracy, I used `GridSearchCV` to perform a search for the best hyperparameters for the model. Due to the time required, I only selected one parameter, `max_depth`, which controls the maximum depth for each `DecisionTreeClassifier` in the forest, in order to prevent overfitting. 

Source code:

```

model = Pipeline([
    ('ColumnDrop', Dropper(['side', 'gameid', 'position', 'league', 'teamname'])),
    ('TextEncoding', encoder),
    ('Model', RandomForestClassifier(random_state=42))
])

hyper_param = {
    'Model__max_depth': list(range(5, 15))
}

cv = GridSearchCV(model, hyper_param, cv=5)
cv.fit(X_train, y_train)

```

As a result, we were able to produce a score of ~0.75 for the testing set (again), and ~0.73 for the Worlds set. While this is indeed an improvement, this is certainly not the best model to use when selecting pick'em teams.

### Final Model
We have to keep in mind that the teams playing in the Worlds series are the best in the world, and their versatilities and decision making can be game changing. Thus the teams themselves play a huge factor in their competition. Hence for my final model, I included the one-hot-encoded teamnames into our dataset.

Source code:
```

# Define column transformer for encoding and dropping columns
encoder = ColumnTransformer(
    transformers=[
        ('OneHot', OneHotEncoder(), ['teamname'])
    ], 
    remainder='passthrough'
)

model_best = Pipeline([
    ('ColumnDrop', Dropper(['side', 'gameid', 'position', 'league'])),
    ('TextEncoding', encoder),
    ('Model', RandomForestClassifier(random_state=42))
])

```
Finally, despite obtaining a fairly low score of test set (contains mostly non-world championship games) of around ~0.75, we are able to bump up the accuracy for prediction of championship games to ~0.95! I cannot wait for the next Worlds to use this for my pick'em selection!
