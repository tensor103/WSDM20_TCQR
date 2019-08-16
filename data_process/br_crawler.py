import operator

import html2text
from bs4 import BeautifulSoup
from github import Github
import re
from tqdm import tqdm
import requests
import time
import random
import os
from os import path
import pandas as pd
from datetime import datetime
import sys
sys.path.append("./")
from br_utils import split

GITHUB_TOKEN = "f0e9d3073c07fc18b8efc762d391659401006118"
# PROJECT = "CAMEL"
PROJECT = "COLLECTIONS"
# PROJECT = "COMPRESS"

REPO_DICT = {
    "CAMEL": "apache/camel",
    "HIVE": "apache/hive",
    "COLLECTIONS": "apache/commons-collections",
    "COMPRESS": "apache/commons-compress"
}

DATA_ROOT = "./data/"
PROJECT_PATH = DATA_ROOT + PROJECT + "/"
ISSUE_PATH = PROJECT_PATH + "issues/"
JIRA_URL_HEADER = "https://issues.apache.org/jira/si/jira.issueviews:issue-xml/"
ASSIGN_FILE = PROJECT_PATH + "assignee.txt"

TRAIN_FILE = PROJECT_PATH + "train.pkl"
VAL_FILE = PROJECT_PATH + "val.pkl"
TEST_FILE = PROJECT_PATH + "test.pkl"

# time cross training
TRAIN_CROSS_FILE = PROJECT_PATH + "train_cross.pkl"
VAL_CROSS_FILE = PROJECT_PATH + "val_cross.pkl"
TEST_CROSS_FILE = PROJECT_PATH + "test_cross.pkl"

# time single training
TRAIN_SINGLE_FILE = PROJECT_PATH + "train_single.pkl"
VAL_SINGLE_FILE = PROJECT_PATH + "val_single.pkl"
TEST_SINGLE_FILE = PROJECT_PATH + "test_single.pkl"

ASSIGNEE_ID_FILE = PROJECT_PATH + "user_ids.txt"


def print_all_devs(usr_dict):
    sorted_x = sorted(usr_dict.items(), key=lambda kv: kv[1], reverse=True)
    for x in sorted_x:
        print(x[0], '\t', x[1], sep='')
    print(sorted_x)

def crawl_assignee(token, repo_name, project_name):

    # First create a Github instance:
    # using token
    g = Github(token)

    repo = g.get_repo(repo_name)
    commits = repo.get_commits()

    usr_dict = {}
    issue_author_dict = {}

    assign_file = open(ASSIGN_FILE, 'w+', encoding='utf-8')
    for commit in tqdm(commits):

        commit_msg = commit.commit.message
        # # filter revert
        # if commit_msg[:6] == "Revert":
        #     #print(commit_msg)
        #     continue

        # extract the issue number
        try:
            #id = re.search(r'^[*' + project_name + r'-(\d+)', commit_msg).group(1)
            id = int(re.search(project_name + r'-(\d+)', commit_msg).group(1))
            pass
        except:
            # print("ID Parsing Error", commit_msg)
            continue

        # get user name
        if commit.committer is None:
            usr_name = commit.commit.committer.name
        else:
            usr_name = commit.committer.login

        if usr_name not in usr_dict:
            usr_dict[usr_name] = 1
        else:
            usr_dict[usr_name] += 1

        if id in issue_author_dict:
            issue_author_dict[id].add(usr_name)
            #print("Duplicate id:", id, "Msg:", commit_msg)
        else:
            issue_author_dict[id] = set([usr_name])


        #print(id, "\t", usr_name, sep="")

        #print(commit.commit.message[:10])
        #print("----")

    sorted_issues = sorted(issue_author_dict.items(), key=operator.itemgetter(0))
    for issue in sorted_issues:
        issue_id, authors = issue
        authors_str = ",".join(authors)
        assign_file.write(str(issue_id) + "\t" + authors_str + "\n")
    assign_file.close()


def download_issue_xml_files(issue_path):

    with open(ASSIGN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            issue_id, usr = line.strip().split('\t')

            url = JIRA_URL_HEADER + PROJECT + "-" + issue_id + "/" + PROJECT + "-" + issue_id + ".xml"
            # wait random time to avoid crawl trapper
            # sleep_time = random.uniform(0, 1)
            # time.sleep(sleep_time)
            response = requests.get(url)

            # output
            output_file = issue_path + PROJECT + "-" + issue_id + ".xml"
            with open(output_file, 'wb') as file:
                file.write(response.content)
            print(issue_id, usr)
    # issues = repo.get_issues()
    # for i, issue in enumerate(issues):
    #     print(i, issue.title)



def parse_issue(br_file):

    # load XML file
    with open(br_file, 'r', encoding='utf-8') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'xml')

        # link
        # content_link = soup.find_all('link')[1].get_text()
        # print(content_link)

        # title and description
        title = soup.find_all('title')[1].get_text()
        # print("Title:", title)
        # print("****************************")
        content_desc = soup.find_all('description')
        formatted_desc = html2text.html2text(content_desc[1].get_text())
        # print(formatted_desc)
        # print("****************************")

        # time

        create_time = soup.find_all('created')[0].get_text() #Thu, 20 Sep 2018 23:46:28 +0000
        create_time = parse_date(create_time)


        updated_time = soup.find_all('updated')[0].get_text()
        updated_time = parse_date(updated_time)
        #resovled_time = soup.find_all('resolved')[0].get_text()


        # Type
        # type = soup.find_all('type')[0].get_text()
        # type_set.add(type)
        # print(type, ":", content_link)

        # assignee
        # assignee = soup.find_all('assignee')[0].get_text()
        # if assignee != "Unassigned":
        #     print(assignee, "|", content_link)
        #     assignee_list.append(assignee)

        # status
        # status = soup.find_all('status')[0].get_text()
        # status_set.add(status)
        # print(status, ":", content_link)

        # priority
        # priority = soup.find_all('priority')[0].get_text()
        # priority_set.add(priority)
        # print(priority, ":", content_link)

        # component
        component_list = []
        components = soup.find_all('component')
        for component in components:
            component = component.get_text()
            # component_set.add(component)
            component_list.append(component)
            # print(component, ":", content_link)

        # print(type)
        # comments
        comment_list, author_list = [], []
        comments = soup.find_all('comment')
        # print(len(comments))
        for i, comment in enumerate(comments):
            # print("\n======== Comment", i, comment["author"], "|", comment["created"], "========")

            # author
            author = comment["author"]
            author_list.append(author)

            # comment
            formatted_comment = html2text.html2text(comment.get_text())
            comment_list.append(formatted_comment)

    return title, formatted_desc, component_list, comment_list, author_list, create_time, updated_time


def generate_pickle(train_ratio=0.7, val_ratio=0.1):

    print("Generate pickle data files ...")

    num_issues = sum(1 for line in open(ASSIGN_FILE))
    num_train = int(num_issues * train_ratio)
    num_val = int(num_issues * val_ratio)

    assignee_id_dict = {}
    num_assignee_not_in_br = 0
    with open(ASSIGN_FILE, 'r', encoding='utf-8') as f:

        rows = []
        for i, line in tqdm(enumerate(f)):
            issue_id, assignee = line.strip().split('\t')

            issue_dict = {}

            # if i >= 100:
            #     break

            # parse issue xml file
            try:
                title, desc, component_list, comment_list, author_list, create_time, updated_time = parse_issue(ISSUE_PATH + PROJECT + "-" + issue_id + ".xml")
            except:
                continue

            for usr in author_list + [assignee]:
                if usr not in assignee_id_dict:
                    idx = len(assignee_id_dict.keys())
                    assignee_id_dict[usr] = idx

            author_list = [assignee_id_dict[i] for i in author_list]

            issue_dict['id'] = issue_id
            issue_dict['title'] = title
            issue_dict['desc'] = desc
            issue_dict['component_list'] = component_list
            issue_dict['comment_list'] = comment_list
            issue_dict['author_list'] = author_list
            issue_dict['assignee'] = assignee_id_dict[assignee]
            issue_dict['create_time'] = create_time
            issue_dict['updated_time'] = updated_time
            rows.append(issue_dict)

            if assignee_id_dict[assignee] not in author_list:
                num_assignee_not_in_br += 1

        train_df = pd.DataFrame(rows[:num_train])
        val_df = pd.DataFrame(rows[num_train:num_train + num_val])
        test_df = pd.DataFrame(rows[num_train + num_val:])

        # save data to data frame file
        train_df.to_pickle(TRAIN_FILE)
        val_df.to_pickle(VAL_FILE)
        test_df.to_pickle(TEST_FILE)

        # generate time cross data
        split_sets = list(split(range(len(rows)), 10))
        train_cross_rows, val_cross_rows, test_cross_rows = [], [], []
        for i, one_split in enumerate(split_sets):

            train_size = int(train_ratio * len(list(one_split)))
            val_size = int(val_ratio * len(list(one_split)))
            train_cross_rows += [rows[j] for j in list(one_split)[:train_size]]
            val_cross_rows += [rows[j] for j in list(one_split)[train_size:train_size + val_size]]
            test_cross_rows += [rows[j] for j in list(one_split)[train_size + val_size:]]

        train_cross_df = pd.DataFrame(train_cross_rows)
        val_cross_df = pd.DataFrame(val_cross_rows)
        test_cross_df = pd.DataFrame(test_cross_rows)
        train_cross_df.to_pickle(TRAIN_CROSS_FILE)
        val_cross_df.to_pickle(VAL_CROSS_FILE)
        test_cross_df.to_pickle(TEST_CROSS_FILE)

        # generate single time data
        # single_split = split_sets[-1]
        # train_size = int(train_ratio * len(list(single_split)))
        # train_single_rows = [rows[j] for j in list(single_split)[:train_size]]
        # test_single_rows = [rows[j] for j in list(single_split)[train_size:]]
        # train_single_df = pd.DataFrame(train_single_rows)
        # test_single_df = pd.DataFrame(test_single_rows)
        # train_single_df.to_pickle(TRAIN_SINGLE_FILE)
        # test_single_df.to_pickle(TEST_SINGLE_FILE)

    print("Number of assignees not in bug report:", num_assignee_not_in_br)
    # write the user
    assign_id_file = open(ASSIGNEE_ID_FILE, 'w')
    sorted_assign_ids = sorted(assignee_id_dict.items(), key=operator.itemgetter(1))
    for i in sorted_assign_ids:
        assign_id_file.write(str(i[0]) + "\t" + str(i[1]) + "\n")


def main():

    print("Start to crawl project", PROJECT)
    # check the project directory
    if not os.path.exists(PROJECT_PATH):
        os.mkdir(PROJECT_PATH)
    # else:
    #     print("Project", PROJECT, "exists ... stop")
    #     return

    # Step 1. crawl the bug report dataset
    if not os.path.exists(ASSIGN_FILE):
        crawl_assignee(GITHUB_TOKEN, REPO_DICT[PROJECT], PROJECT)

    # Step 2. download issue xml files
    if not os.path.exists(ISSUE_PATH):
        os.mkdir(ISSUE_PATH)
        download_issue_xml_files(ISSUE_PATH)

    # Step 3. preprocessing and save to csv file
    # Format:
    # <BR ID> <BR Title> <BR Description> <List of comments> <List of authors>
    if not (os.path.exists(TRAIN_FILE) or os.path.exists(TEST_FILE)):
        generate_pickle(train_ratio=0.6, val_ratio=0.1)

def parse_date(date_str):
    # sample: Thu, 20 Sep 2018 23:46:28 +0000
    return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')

if __name__ == "__main__":
    main()

