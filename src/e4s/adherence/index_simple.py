import argparse
from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig


def index_documents(collection, index_name, experiment, checkpoint, doc_maxlen, nbits):
    with Run().context(RunConfig(nranks=1, experiment=experiment)):
        config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            checkpoint=checkpoint,
        )
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', required=True)
    parser.add_argument('--index_name', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--checkpoint', default='colbert-ir/colbertv2.0')
    parser.add_argument('--doc_maxlen', type=int, default=512)
    parser.add_argument('--nbits', type=int, default=8)
    args = parser.parse_args()

    index_documents(
        collection=args.collection,
        index_name=args.index_name,
        experiment=args.experiment,
        checkpoint=args.checkpoint,
        doc_maxlen=args.doc_maxlen,
        nbits=args.nbits
    )
