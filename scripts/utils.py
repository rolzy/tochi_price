import numpy as np
import pandas as pd
import lightgbm as lgb 
from sklearn.model_selection import KFold, train_test_split 
from sklearn.metrics import accuracy_score

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def basic_preprocessing():
    tochi_train = pd.read_csv('../data/train_genba.tsv', sep='\t')
    build_train = pd.read_csv('../data/train_goto.tsv', sep='\t')
    train = pd.merge(tochi_train, build_train, on="pj_no")

    tochi_test = pd.read_csv('../data/test_genba.tsv', sep='\t')
    build_test = pd.read_csv('../data/test_goto.tsv', sep='\t')
    test = pd.merge(tochi_test, build_test, on="pj_no")

    submission = pd.DataFrame()
    submission['id'] = test['id']

    # 名前系と相関性が高いカラムはとりあえず削除
    # 土地のネームバリューが出てくると思うので、名前系はあとで追加するかも

    name_columns = ['bastei_nm1','bastei_nm2','chiseki_kb_hb','eki_nm1','eki_nm2','gk_chu_tm','gk_sho_tm', \
                    'hy1f_date_su', 'hy2f_date_su','mseki_yt_hb','tc_mseki','yoseki2','id']

    train.drop(name_columns, axis=1, inplace=True)
    test.drop(name_columns, axis=1, inplace=True)

    train['fi3m_yohi'].replace('（無）','（不要）',inplace=True)
    train['hiatari'].fillna('普通', inplace=True)
    train['kborjs'].replace('公募','公簿',inplace=True)
    test['fi3m_yohi'].replace('（無）','（不要）',inplace=True)
    test['hiatari'].fillna('普通', inplace=True)
    test['kborjs'].replace('公募','公簿',inplace=True)

    maru_columns = ['rs_e_kdate2','rs_e_kdate3','rs_e_m_ari','rs_e_m_nashi','rs_e_parking','rs_e_tahata', \
                    'rs_e_zoki', 'rs_n_kdate2','rs_n_kdate3','rs_n_m_ari','rs_n_m_nashi','rs_n_parking', \
                    'rs_n_tahata','rs_n_zoki', 'rs_s_kdate2','rs_s_kdate3','rs_s_m_ari','rs_s_m_nashi', \
                    'rs_s_parking','rs_s_tahata','rs_s_zoki', 'rs_w_kdate2','rs_w_kdate3','rs_w_m_ari', \
                    'rs_w_m_nashi','rs_w_parking','rs_w_tahata','rs_w_zoki', 'sho_conv','sho_market', \
                    'sho_shoten','sho_super','shu_bochi','shu_factory','shu_highway', 'shu_hvline', \
                    'shu_jutaku','shu_kaido','shu_kokyo','shu_line_ari','shu_line_nashi','shu_park', \
                    'shu_shop','shu_sogi','shu_soon','shu_tower','shu_zoki']

    train[maru_columns] = train[maru_columns].replace({'○':1, np.nan:0})
    test[maru_columns] = test[maru_columns].replace({'○':1, np.nan:0})

    # 他規制や個別要因など、「複数ある場合は1～4」系のカラムに対処

    hokakisei=['hokakisei1','hokakisei2','hokakisei3','hokakisei4']
    kobetsu=['kobetsu1','kobetsu2','kobetsu3','kobetsu4']

    train = pd.concat([train, train[hokakisei].stack().str.get_dummies().sum(level=0), \
                            train[kobetsu].stack().str.get_dummies().sum(level=0)], axis=1)
    train.drop(hokakisei+kobetsu, axis=1, inplace=True)
    train.iloc[:,136:] = train.iloc[:,136:].fillna(0.0).apply(lambda x: [0 if y == 0.0 else 1 for y in x])
    test = pd.concat([test, test[hokakisei].stack().str.get_dummies().sum(level=0), \
                            test[kobetsu].stack().str.get_dummies().sum(level=0)], axis=1)
    test.drop(hokakisei+kobetsu, axis=1, inplace=True)
    test.iloc[:,135:] = test.iloc[:,135:].fillna(0.0).apply(lambda x: [0 if y == 0.0 else 1 for y in x])

    # BooleanであるハズがCategoricalになってるカラムに対処

    bool_columns = ['bus_yohi','chikukeikaku','fi3m_yohi','fi4m_yohi','gesui','hokakyoka','josui','kaihatsukyoka', \
                    'kaoku_um', 'kborjs','keikakuroad','kinshijiko','t53kyoka','yheki_umu','yheki_yohi']

    train[bool_columns] = train[bool_columns].replace({'（不要）':0, '（無）':0,'（要）':1,'（有）':1, \
                                        '公共下水':0,'個別浄化槽':1,'公営':0,'私営':1,'実測':0,'公簿':1})
    test[bool_columns] = test[bool_columns].replace({'（不要）':0, '（無）':0,'（要）':1,'（有）':1,
                                        '公共下水':0,'個別浄化槽':1,'公営':0,'私営':1,'実測':0,'公簿':1})

    # ちょっとデータを追加
    # Levelplanから階数と部屋を分割
    # 公表された平均価格の平均を追加

    levelplan_split_train = train['levelplan'].str.split('/', n=1, expand=True)
    train['level'] = levelplan_split_train[0]
    train['rooms'] = levelplan_split_train[1]

    levelplan_split_test = test['levelplan'].str.split('/', n=1, expand=True)
    test['level'] = levelplan_split_test[0]
    test['rooms'] = levelplan_split_test[1]

    # 全部埼玉県なので住居から消去
    # あと市が抜けてる住所情報は追加してあげる

    shi_gun_dic = dict({'にっさい花みず木':'坂戸市にっさい花みず木','西鶴ヶ岡':'ふじみ野市西鶴ヶ岡', \
                        '杉戸町内田':'北葛飾郡杉戸町内田','宮代町宮代台':'南埼玉郡宮代町宮代台', \
                        '大字下日出谷':'桶川市大字下日出谷','杉戸町清地':'北葛飾郡杉戸町', \
                        '松伏町田中':'北葛飾郡松伏町','大字水野字逃水':'狭山市大字水野字逃水'})

    train['jukyo'] = train['jukyo'].str.replace('埼玉県','')
    test['jukyo'] = test['jukyo'].str.replace('埼玉県','')
    train['jukyo'] = train['jukyo'].replace(shi_gun_dic)
    test['jukyo'] = test['jukyo'].replace(shi_gun_dic)

    jukyo_split_train = train['jukyo'].str.split(r'市|郡', n=1, expand=True)
    train['jukyo_shi_gun'] = jukyo_split_train[0]
    train.drop('jukyo', axis=1, inplace=True)

    jukyo_split_test = test['jukyo'].str.split(r'市|郡', n=1, expand=True)
    test['jukyo_shi_gun'] = jukyo_split_test[0]
    test.drop('jukyo', axis=1, inplace=True)

    # 最後に、categoricalなカラムを全てone_hot_encode
    categorical = ['bas_toho1','bas_toho2','bokachiiki','gas','hiatari','hw_status','jigata','kodochiku', \
                   'levelplan', 'road1_hk','road1_sb','road2_hk','road2_sb','road3_sb','road3_hk','road4_sb', \
                   'road4_hk','road_st', 'rosen_nm1','rosen_nm2','setsudo_hi','setsudo_kj','toshikuiki1', \
                   'toshikuiki2','usui','yoto1','yoto2']

    train = pd.concat([train, pd.get_dummies(train[categorical])], axis=1)
    train.drop(categorical, axis=1, inplace=True)
    test = pd.concat([test, pd.get_dummies(test[categorical])], axis=1)
    test.drop(categorical, axis=1, inplace=True)

    # 両方のDataframeに登場しないカラムを除外（価格はキープしとく）

    train_columns = list(train.columns.values)
    test_columns = list(test.columns.values)
    unique_columns = list(set(train_columns) ^ set(test_columns))

    y_train = train['keiyaku_pr']
    train.drop(unique_columns, axis=1, inplace=True, errors='ignore')
    test.drop(unique_columns, axis=1, inplace=True, errors='ignore')

    return train, test, y_train, submission

def lightgbm_5fold(train, y_train):
    # 3分割交差検証を指定し、インスタンス化 
    kf = KFold(n_splits=5) 

    # スコアとモデルを格納するリスト 
    score_list = [] 
    models = [] 

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    for fold_, (train_index, valid_index) in enumerate(kf.split(train, y_train)):
        train_x = train.iloc[train_index]
        valid_x = train.iloc[valid_index]
        train_y = y_train[train_index]
        valid_y = y_train[valid_index]

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_eval = lgb.Dataset(valid_x, valid_y, reference=lgb_train)
        print(f'fold{fold_ + 1} start')
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=5000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=20,
                        verbose_eval=0)
        y_pred = gbm.predict(valid_x, num_iteration=gbm.best_iteration)
        score_list.append(mean_absolute_percentage_error(valid_y, y_pred))
        models.append(gbm)  # 学習が終わったモデルをリストに入れておく
        print(f'fold{fold_ + 1} end\nAccuracy = {mean_absolute_percentage_error(valid_y, y_pred)}')
    print(score_list, '平均score', np.mean(score_list))

    return models

def predict_5fold(test, models, submission):
    test_pred = np.zeros((len(test), 5)) 
    for fold_, gbm in enumerate(models):
        pred_ = gbm.predict(test, num_iteration=gbm.best_iteration)# testを予測
        test_pred[:, fold_] = pred_ 
    pred = np.mean(test_pred, axis=1)
    submission['price'] = pred
    submission.to_csv('lightgbm.tsv', sep='\t', index=False, header=False)
