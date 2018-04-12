from abc import ABCMeta, abstractmethod
import datetime
import psycopg2
from pymongo import MongoClient


class ModelWriter:
    """
    Abstract class for a model store writer.
    Implement backend specific writers as a subclass.
    """
    __metaclass__ = ABCMeta

    def __init__(self, sc, uri):
        """

        :param sc: A Spark context
        :param uri: The connection URI
        """
        self._sc = sc
        self._url = uri

    @abstractmethod
    def write(self, model, version):
        """
        Writes a specific `model` with unique `version` to the model store.

        :param model: A instance of a Spark ALS `MatrixFactorizationModel`
        :param version: The (unique) `model`'s version
        """
        pass


class MongoDBModelWriter(ModelWriter):
    """
    Model store writer to a MongoDB backend
    """

    def __init__(self, sc, uri):
        super(MongoDBModelWriter, self).__init__(sc=sc, uri=uri)
        client = MongoClient(self._url)
        self._db = client.models

    def write(self, model, version):

        data = {'id': version,
                'rank': model.rank,
                'created': datetime.datetime.utcnow()}

        self._db.models.insert_one(data)

        u = model.userFeatures().collect()

        for feature in u:
            self._db.userFactors.insert_one({
                'model_id': version,
                'id': feature[0],
                'features': list(feature[1])})

        p = model.productFeatures().collect()

        for feature in p:
            self._db.productFactors.insert_one({
                'model_id': version,
                'id': feature[0],
                'features': list(feature[1])})


class DataLoader:
    """
    Abstract class for a Data Store loader.
    Implement backend specific loaders as a subclass.
    """
    __metaclass__ = ABCMeta

    def __init__(self, arguments):
        """
        :param arguments: The database specific connection arguments
        """
        self._arguments = arguments

    @abstractmethod
    def fetchall(self):
        """
        Returns all ratings in the database. Each rating must be in form
        `(userid, productid, rating) and in a collection type that Spark's
        `parallelize` can accept (e.g. `List`)
        :return: A collection of ratings, each with a user, product and rating.
        """
        pass

    @abstractmethod
    def latest_timestamp(self):
        """
        Returns the timestamp for the most recent (chronologically)
        rating in the database.
        :return: A timestamp
        """
        pass

    @abstractmethod
    def fetchafter(self, timestamp):
        """
        Returns all ratings (in the same format as `fetchall`) added after
        `timestamp`.
        :param timestamp: A timestamp
        :return: A collection of ratings, each with a user, product and rating.
        """
        pass


class PostgresDataLoader(DataLoader):
    """
    Data store loader for a PostgreSQL backend.
    """

    def __init__(self, arguments):
        super(PostgresDataLoader, self).__init__(arguments)
        self._connection = self._build_connection(arguments)
        self._cursor = self._connection.cursor()

    def _make_connection(self, host='127.0.0.1', port=5432, user='postgres',
                         password='postgres', dbname='postgres'):
        """Connect to a postgresql db."""
        return psycopg2.connect(host=host, port=port, user=user,
                                password=password, dbname=dbname)

    def _build_connection(self, args):
        """Make the db connection with an args object."""
        conn = self._make_connection(host=args.host,
                                     port=args.port,
                                     user=args.user,
                                     password=args.password,
                                     dbname=args.dbname)
        return conn

    def fetchall(self):
        self._cursor.execute("SELECT * FROM ratings")
        return self._cursor.fetchall()

    def latest_timestamp(self):
        self._cursor.execute(
            "SELECT timestamp FROM ratings ORDER BY timestamp DESC LIMIT 1;"
        )
        return self._cursor.fetchone()[0]

    def fetchafter(self, timestamp):
        self._cursor.execute(
            "SELECT * FROM ratings WHERE (timestamp > %s);",
            (timestamp,))
        return self._cursor.fetchall()
