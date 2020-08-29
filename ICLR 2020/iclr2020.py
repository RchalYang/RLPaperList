

import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm
import openreview


def download_iclr20(client, outdir='./', get_pdfs=False):
    '''
    Main function for downloading ICLR metadata (and optionally, PDFs)
    '''
    # pylint: disable=too-many-locals

    id_to_submission = {
        note.id: note for note in openreview.tools.iterget_notes(client, invitation = 'ICLR.cc/2020/Conference/-/Blind_Submission')
    }
    decisions = openreview.tools.iterget_notes(
        client, invitation='ICLR.cc/2020/Conference/Paper.*/-/Decision')
    decisions = list(decisions)
    reviews = openreview.tools.iterget_notes(
        client, invitation='ICLR.cc/2020/Conference/Paper.*/-/Official_Review')

    reviews_by_forum = defaultdict(list)
    for review in reviews:
        reviews_by_forum[review.forum].append(review)

    accepted_submissions = [id_to_submission[note.forum] for note in decisions if 'Accept' in note.content['decision']]


    accepted_by_forum = {n.forum: n for n in accepted_submissions}
    decision_by_forum = {n.forum: n for n in decisions}
    print(len(decision_by_forum))

    metadata = []
    for forum in accepted_by_forum:

        forum_reviews = reviews_by_forum[forum]

        review_ratings = [n.content['rating'] for n in forum_reviews]

        decision = decision_by_forum[forum].content['decision'][8:-1]

        submission_content = accepted_by_forum[forum].content

        forum_metadata = {
            'forum': forum,
            'review_ratings': review_ratings,
            'decision': decision,
            'content': submission_content
        }
        metadata.append(forum_metadata)

    rl_submissions = {"Spotlight": [],
                      "Poster": [],
                       "Talk": []}

    keywords = defaultdict(int)

    for forum_metadata in metadata:
        flag = False
        for keyword in forum_metadata['content']['keywords']:
            if "reinforcement learning" in keyword or "rl" == keyword:
                rl_submissions[forum_metadata["decision"]].append(forum_metadata)
                flag = True
                break
        if flag:
            # keywords += keyword
            for keyword in forum_metadata['content']['keywords']:
                keywords[keyword] += 1

    keywords = [(k, v) for k, v in sorted(keywords.items(), key=lambda item: item[1])]
    keywords.reverse()
    print(keywords)

    with open(os.path.join(outdir, 'Statistics.md'), 'w') as file_handle:
        file_handle.write("# Statistics\n")
        file_handle.write("\n")
        file_handle.write("## Counts \n")
        file_handle.write("\n")
        total = 0
        for cat in rl_submissions:
            total += len(rl_submissions[cat])
            # file_handle.write(json.dumps(forum_metadata) + '\n')
            file_handle.write("{} count: {}\n".format(cat, len(rl_submissions[cat])))
            file_handle.write("\n")
        
        file_handle.write("{} count: {}\n".format("Total", total ))
        file_handle.write("\n")

        file_handle.write("## Keywords\n")
        file_handle.write("\n")
        for keyword in keywords:
            file_handle.write("{}:{}\n\n".format(keyword[0], keyword[1]))


    for cat in rl_submissions.keys():
        with open(os.path.join(outdir, '{}.md'.format(cat)), 'w') as file_handle:
            file_handle.write("# {}\n".format(cat))
            file_handle.write("\n")
            for forum_metadata in rl_submissions[cat]:
                # file_handle.write(json.dumps(forum_metadata) + '\n')
                file_handle.write("## {}\n".format(forum_metadata['content']['title']))
                file_handle.write("\n")
                file_handle.write("Authors: {}\n".format(forum_metadata['content']['authors']))
                file_handle.write("\n")
                file_handle.write("Ratings: {}\n".format(sorted(forum_metadata['review_ratings'])))
                file_handle.write("\n")
                file_handle.write("Keywords: {}\n".format(forum_metadata['content']['keywords']))
                file_handle.write("\n")
                file_handle.write("{}\n".format(forum_metadata['content']['abstract']))
                file_handle.write("\n")

        with open(os.path.join(outdir, 'iclr20_rl_{}.json'.format(cat)), 'w') as file_handle:
            for forum_metadata in rl_submissions[cat]:
                file_handle.write(json.dumps(forum_metadata) + '\n')

    # if requested, download pdfs to a subdirectory.
    if get_pdfs:

        base_pdf_outdir = os.path.join(outdir, 'iclr20_pdfs_rl')
        os.makedirs(base_pdf_outdir)
        for cat in rl_submissions.keys():
            pdf_outdir = os.path.join(base_pdf_outdir, cat)
            os.makedirs(pdf_outdir)
            for forum_metadata in tqdm(rl_submissions[cat], desc='getting pdfs'):
                pdf_binary = client.get_pdf(forum_metadata['forum'])
                pdf_outfile = os.path.join(pdf_outdir, '{}.pdf'.format(forum_metadata['content']['title']))
                with open(pdf_outfile, 'wb') as file_handle:
                    file_handle.write(pdf_binary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--outdir', default='./', help='directory where data should be saved')
    parser.add_argument(
        '--get_pdfs', default=False, action='store_true', help='if included, download pdfs')
    parser.add_argument('--baseurl', default='https://openreview.net')
    parser.add_argument('--username', default='', help='defaults to empty string (guest user)')
    parser.add_argument('--password', default='', help='defaults to empty string (guest user)')

    args = parser.parse_args()

    outdir = args.outdir

    client = openreview.Client(
        baseurl=args.baseurl,
        username=args.username,
        password=args.password)

    download_iclr20(client, outdir, get_pdfs=args.get_pdfs)

