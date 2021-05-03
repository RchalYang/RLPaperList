

import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm
import openreview


def download_iclr21(client, outdir='./'):
    '''
    Main function for downloading ICLR metadata (and optionally, PDFs)
    '''
    # pylint: disable=too-many-locals

    rl_submissions = []

    keywords = defaultdict(int)

    for submission in openreview.tools.iterget_notes(
            client, invitation = 'ICLR.cc/2021/Conference/-/Blind_Submission'):
        # keywords = [ keyword.lower() for keyword in submission["content"]["keywors"]]
        # print(submission)
        # print(submission.content)
        if "keywords" not in submission.content:
            continue
        flag = False
        for keyword in submission.content["keywords"]:
            if "reinforcement learning" in keyword.lower() or "rl" == keyword.lower():
                rl_submissions.append(submission)
                flag = True
                break
        if flag:
            # keywords += keyword
            for keyword in submission.content['keywords']:
                keywords[keyword] += 1

    keywords = [(k, v) for k, v in sorted(keywords.items(), key=lambda item: item[1])]
    keywords.reverse()
    # print(keywords)

    with open(os.path.join(outdir, 'Statistics.md'), 'w') as file_handle:
        file_handle.write("# Statistics\n")
        file_handle.write("\n")
        file_handle.write("## Counts \n")
        file_handle.write("\n")
        total = 0
        
        file_handle.write("{} count: {}\n".format("Total", len(rl_submissions)))
        file_handle.write("\n")

        file_handle.write("## Keywords\n")
        file_handle.write("\n")
        for keyword in keywords:
            file_handle.write("{}:{}\n\n".format(keyword[0], keyword[1]))


    with open(os.path.join(outdir, 'Submissions.md'), 'w') as file_handle:
        file_handle.write("# Submissions\n")
        file_handle.write("\n")
        for submission in rl_submissions:
            # print(submission)
            file_handle.write("## {}\n".format(submission.content['title']))
            file_handle.write("\n")
            file_handle.write("Authors: {}\n".format(submission.content['authors']))
            file_handle.write("\n")
            file_handle.write("Keywords: {}\n".format(submission.content['keywords']))
            file_handle.write("\n")
            file_handle.write("{}\n".format(submission.content['abstract']))
            file_handle.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--outdir', default='./', help='directory where data should be saved')
    parser.add_argument('--baseurl', default='https://api.openreview.net')
    parser.add_argument('--username', default='', help='defaults to empty string (guest user)')
    parser.add_argument('--password', default='', help='defaults to empty string (guest user)')

    args = parser.parse_args()

    outdir = args.outdir

    client = openreview.Client(
        baseurl=args.baseurl,
        username=args.username,
        password=args.password)

    download_iclr21(client, outdir)

