import warnings
from urllib.parse import quote_plus

import pandas as pd
import pyodbc
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")


class CescoRodbConnection:
    def __init__(self):
        self.server = "rodb.cesco.biz,11433"
        self.database = "CESCOEIS"
        self.username = "CX_SLLM"
        self.password = "Cesco@1588"
        self.available_drivers = [
            "ODBC Driver 17 for SQL Server",
            "ODBC Driver 13 for SQL Server",
            "SQL Server Native Client 11.0",
            "SQL Server",
        ]
        self.connection = None
        self.engine = None
        self.working_driver = None

    def get_available_drivers(self):
        """시스템에서 사용 가능한 ODBC 드라이버 확인"""
        available = pyodbc.drivers()

        for driver in self.available_drivers:
            if driver in available:
                return driver

        if available:
            return available[0]

        print("❌ 사용 가능한 ODBC 드라이버를 찾을 수 없습니다!")
        return None

    def connect(self):
        # 사용 가능한 드라이버 찾기
        self.working_driver = self.get_available_drivers()
        if not self.working_driver:
            return False

        try:
            # 연결 문자열 생성
            connection_string = (
                f"DRIVER={{{self.working_driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
                f"TrustServerCertificate=yes;"
                f"Encrypt=no;"
            )

            # Direct pyodbc connection
            self.connection = pyodbc.connect(connection_string, timeout=30)

            # SQLAlchemy engine
            self.engine = create_engine(
                f"mssql+pyodbc:///?odbc_connect={quote_plus(connection_string)}",
                fast_executemany=True,
                pool_timeout=30,
                pool_recycle=3600,
            )
            return True

        except Exception as e:
            print(f"❌ 연결 실패: {e}")
            print("💡 해결 방법:")
            print("   1. VPN 연결 확인")
            print("   2. 서버 주소 확인: rodb.cesco.biz,11433")
            print("   3. 방화벽 설정 확인")
            print("   4. ODBC 드라이버 설치 확인")
            return False

    def test_connection(self):
        if not self.engine:
            return False

        try:
            test_query = "SELECT COUNT(*) as total FROM CESCOEIS.dbo.TB_고객"
            result = pd.read_sql_query(test_query, self.engine)
            return True
        except Exception as e:
            print(f"❌ 연결 테스트 실패: {e}")
            return False

    def close(self):
        """연결 종료"""
        try:
            if self.connection:
                self.connection.close()
                print("🔒 RODB pyodbc 연결 종료")
            if self.engine:
                self.engine.dispose()
                print("🔒 RODB SQLAlchemy 엔진 종료")
        except Exception as e:
            print(f"⚠️ RODB 연결 종료 중 오류: {e}")

    def execute_query(self, query, params=None):
        """SQL 쿼리 실행 및 결과 반환"""
        if not self.engine:
            print("🔄 RODB 연결이 설정되지 않았습니다. 새로 연결합니다.")
            self.connect()

        if not self.engine:
            print("❌ RODB 연결을 설정할 수 없습니다.")
            return None

        try:
            # Test if engine is still valid by checking if it's disposed
            if self.engine.pool._is_disposed:
                print("⚠️ RODB 엔진이 disposed 상태입니다. 재연결합니다.")
                self.connect()
                if not self.engine:
                    print("❌ RODB 재연결 실패.")
                    return None
        except AttributeError:
            # Pool might not have _is_disposed attribute, continue
            pass

        try:
            # 연결을 새로고침하거나 테스트하여 유효한지 확인
            with self.engine.connect() as connection:
                result = pd.read_sql_query(query, connection, params=params)
            return result
        except pyodbc.OperationalError as e:
            if "connection is closed" in str(e).lower():
                print("⚠️ RODB 연결이 닫혔습니다. 다시 연결하여 재시도합니다.")
                self.engine = None  # Force recreation
                self.connect()  # Reconnect
                if not self.engine:
                    print("❌ RODB 재연결 실패.")
                    return None
                try:
                    with self.engine.connect() as connection:
                        result = pd.read_sql_query(query, connection, params=params)
                    print(f"✅ RODB 쿼리 재시도 성공! 결과: {len(result)}행")
                    return result
                except Exception as retry_e:
                    print(f"❌ RODB 쿼리 재시도 실패: {retry_e}")
                    print(f"🔍 실행된 쿼리: {query[:200]}...")
                    return None
            else:
                print(f"❌ RODB 쿼리 실행 실패: {e}")
                print(f"🔍 실행된 쿼리: {query[:200]}...")
                return None
        except Exception as e:
            print(f"❌ RODB 쿼리 실행 중 예외 발생: {e}")
            print(f"🔍 실행된 쿼리: {query[:200]}...")
            return None


class CescoCXConnection:
    def __init__(self):
        self.server = "cescobi.cesco.biz,11433"
        self.database = "CX_CDM"
        self.username = "CX_SLLM"
        self.password = "Cesco@1588"
        self.available_drivers = [
            "ODBC Driver 17 for SQL Server",
            "ODBC Driver 13 for SQL Server",
            "SQL Server Native Client 11.0",
            "SQL Server",
        ]
        self.connection = None
        self.engine = None
        self.working_driver = None

    def get_available_drivers(self):
        """시스템에서 사용 가능한 ODBC 드라이버 확인"""
        available = pyodbc.drivers()
        print(f"🚗 시스템 ODBC 드라이버: {available}")

        for driver in self.available_drivers:
            if driver in available:
                print(f"✅ 사용 가능한 드라이버 발견: {driver}")
                return driver

        if available:
            print(
                f"⚠️ 기본 드라이버를 사용하지 못하여 첫 번째 사용 가능한 드라이버 사용: {available[0]}"
            )
            return available[0]

        print("❌ 사용 가능한 ODBC 드라이버를 찾을 수 없습니다!")
        return None

    def connect(self):
        # 사용 가능한 드라이버 찾기
        self.working_driver = self.get_available_drivers()
        if not self.working_driver:
            return False

        try:
            # 연결 문자열 생성
            connection_string = (
                f"DRIVER={{{self.working_driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
                f"TrustServerCertificate=yes;"
                f"Encrypt=yes;"
            )

            # Direct pyodbc connection
            self.connection = pyodbc.connect(connection_string, timeout=30)

            # SQLAlchemy engine
            self.engine = create_engine(
                f"mssql+pyodbc:///?odbc_connect={quote_plus(connection_string)}",
                fast_executemany=True,
                pool_timeout=30,
                pool_recycle=3600,
            )
            return True

        except Exception as e:
            print(f"❌ BIDB/CX 연결 실패: {e}")
            print("💡 해결 방법:")
            print("   1. VPN 연결 확인")
            print("   2. 서버 주소 확인: cescobi.cesco.biz,11433")
            print("   3. 방화벽 설정 확인")
            print("   4. ODBC 드라이버 설치 확인")
            return False

    def test_connection(self):
        if not self.engine:
            return False

        try:
            test_query = "select TOP 1 * from CX_CDM.dbo.DA_M_MYLAB_PROFIT_DAILY"
            result = pd.read_sql_query(test_query, self.engine)
            return True
        except Exception as e:
            print(f"❌ BIDB 연결 테스트 실패: {e}")
            return False

    def close(self):
        """연결 종료"""
        try:
            if self.connection:
                self.connection.close()
                print("🔒 BIDB pyodbc 연결 종료")
            if self.engine:
                self.engine.dispose()
                print("🔒 BIDB SQLAlchemy 엔진 종료")
        except Exception as e:
            print(f"⚠️ BIDB 연결 종료 중 오류: {e}")

    def execute_query(self, query, params=None):
        """SQL 쿼리 실행 및 결과 반환"""
        if not self.engine:
            print("🔄 BIDB/CX 연결이 설정되지 않았습니다. 새로 연결합니다.")
            self.connect()

        if not self.engine:
            print("❌ BIDB/CX 연결을 설정할 수 없습니다.")
            return None

        try:
            # Test if engine is still valid by checking if it's disposed
            if self.engine.pool._is_disposed:
                print("⚠️ BIDB/CX 엔진이 disposed 상태입니다. 재연결합니다.")
                self.connect()
                if not self.engine:
                    print("❌ BIDB/CX 재연결 실패.")
                    return None
        except AttributeError:
            # Pool might not have _is_disposed attribute, continue
            pass

        try:
            # 컨텍스트 관리자를 사용하여 항상 유효한 연결을 얻습니다.
            with self.engine.connect() as connection:
                result = pd.read_sql_query(query, connection, params=params)
            return result
        except pyodbc.OperationalError as e:
            # 연결이 닫혔다는 오류가 발생하면 재연결 후 재시도합니다.
            if (
                "connection is closed" in str(e).lower()
                or "connection does not exist" in str(e).lower()
            ):
                print("⚠️ BIDB/CX 연결이 닫혔습니다. 다시 연결하여 재시도합니다.")
                self.engine = None  # Force recreation
                self.connect()
                if not self.engine:
                    print("❌ BIDB/CX 재연결 실패.")
                    return None
                try:
                    with self.engine.connect() as connection:
                        result = pd.read_sql_query(query, connection, params=params)
                    print(f"✅ BIDB/CX 쿼리 재시도 성공! 결과: {len(result)}행")
                    return result
                except Exception as retry_e:
                    print(f"❌ BIDB/CX 쿼리 재시도 실패: {retry_e}")
                    print(f"🔍 실행된 쿼리: {query[:200]}...")
                    return None
            else:
                print(f"❌ BIDB/CX 쿼리 실행 중 OperationalError: {e}")
                print(f"🔍 실행된 쿼리: {query[:200]}...")
                return None
