import numpy as np
import pandas as pd

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
                    'hy1f_date_su', 'hy2f_date_su','jukyo','mseki_yt_hb','tc_mseki','yoseki2','id']
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
    train.iloc[:,137:] = train.iloc[:,137:].fillna(0.0).apply(lambda x: [0 if y == 0.0 else 1 for y in x])
    test = pd.concat([test, test[hokakisei].stack().str.get_dummies().sum(level=0), \
                            test[kobetsu].stack().str.get_dummies().sum(level=0)], axis=1)
    test.drop(hokakisei+kobetsu, axis=1, inplace=True)
    test.iloc[:,136:] = test.iloc[:,136:].fillna(0.0).apply(lambda x: [0 if y == 0.0 else 1 for y in x])

    # BooleanであるハズがCategoricalになってるカラムに対処

    bool_columns = ['bus_yohi','chikukeikaku','fi3m_yohi','fi4m_yohi','gesui','hokakyoka','josui','kaihatsukyoka', \
                    'kaoku_um', 'kborjs','keikakuroad','kinshijiko','t53kyoka','yheki_umu','yheki_yohi']

    train[bool_columns] = train[bool_columns].replace({'（不要）':0, '（無）':0,'（要）':1,'（有）':1, \
                                        '公共下水':0,'個別浄化槽':1,'公営':0,'私営':1,'実測':0,'公簿':1})
    test[bool_columns] = test[bool_columns].replace({'（不要）':0, '（無）':0,'（要）':1,'（有）':1,
                                        '公共下水':0,'個別浄化槽':1,'公営':0,'私営':1,'実測':0,'公簿':1})

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
