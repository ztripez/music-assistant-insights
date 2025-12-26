#!/usr/bin/env python3
"""
Phase 1 Integration Test
Purpose: Verify taste profile system integration using the actual SidecarClient
Date: 2025-12-26

Tests both storage backends:
1. File-based storage (usearch)
2. Qdrant storage

For each backend:
1. Sidecar startup and health check
2. Text embedding generation
3. Track storage
4. Taste profile computation
"""

import asyncio
import subprocess
import msgpack
import sys

from test_sidecar_client import SidecarClient, TrackMetadata


SIDECAR_URL = "http://localhost:8096"
QDRANT_URL = "http://services.home.vonmatern.org:6334"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
NC = "\033[0m"  # No Color


async def run_tests_with_storage(storage_mode: str, storage_args: list):
    """Run all integration tests with specified storage backend."""
    print()
    print(f"{BLUE}{'=' * 60}{NC}")
    print(f"{BLUE}Testing with {storage_mode.upper()} storage{NC}")
    print(f"{BLUE}{'=' * 60}{NC}")
    print()

    # Start sidecar with specified storage
    print(f"{YELLOW}[1/7] Starting sidecar with {storage_mode} storage...{NC}")
    cmd = ["./target/debug/insight-sidecar"] + storage_args
    print(f"  Command: {' '.join(cmd)}")
    sidecar = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for sidecar to start and storage to be ready
        print("Waiting for sidecar to start...")
        client = SidecarClient(base_url=SIDECAR_URL)

        storage_ready = False
        for i in range(30):
            try:
                health = await client.health_check()
                if health.get("storage_ready", False):
                    storage_ready = True
                    print(f"{GREEN}✓ Sidecar and storage ready{NC}")
                    break
                elif i >= 5:  # Give storage some time to connect
                    print(f"Storage not ready yet... ({i}/30)")
            except Exception:
                pass

            if i == 29:
                print(f"{RED}✗ Sidecar/storage failed to start within 30 seconds{NC}")
                return False
            await asyncio.sleep(1)

        if not storage_ready:
            print(f"{RED}✗ Storage not ready{NC}")
            return False
        print()

        # Test 2: Health endpoint
        print(f"{YELLOW}[2/7] Testing health endpoint...{NC}")
        try:
            health = await client.health_check()
            print(f"Health response: {health}")
            if "status" in health:
                print(f"{GREEN}✓ Health endpoint working{NC}")
            else:
                print(f"{RED}✗ Health endpoint failed{NC}")
                return False
        except Exception as e:
            print(f"{RED}✗ Health endpoint failed: {e}{NC}")
            return False
        print()

        # Test 3: Text embedding generation (raw text to embedding)
        print(f"{YELLOW}[3/7] Testing text embedding generation...{NC}")
        try:
            # Generate embedding for a text query using the search method's embed logic
            session = await client._get_session()
            embed_url = f"{client.base_url}/api/v1/embed/text"
            embed_payload = {"text": "electronic ambient music with dreamy atmosphere"}
            embed_data = msgpack.packb(embed_payload)

            async with session.post(embed_url, data=embed_data) as resp:
                if resp.status != 200:
                    print(f"{RED}✗ Text embedding failed{NC}")
                    return False
                result = msgpack.unpackb(await resp.read(), raw=False)

            if "embedding" in result:
                embedding = result["embedding"]
                dim = len(embedding)
                print(f"{GREEN}✓ Text embedding generated successfully{NC}")
                print(f"Embedding dimension: {dim}")
            else:
                print(f"{RED}✗ Text embedding generation failed{NC}")
                print(f"Response: {result}")
                return False
        except Exception as e:
            print(f"{RED}✗ Text embedding generation failed: {e}{NC}")
            return False
        print()

        # Test 4: Store track embeddings (taste profiles need these)
        print(f"{YELLOW}[4/7] Storing track embeddings for taste profile...{NC}")
        try:
            # Store track-1
            metadata1 = TrackMetadata(
                name="Electronic Dreams",
                artists=["DJ Test"],
                album="Digital",
                genres=["electronic", "techno"],
            )
            await client.embed_text_and_store("track-1", metadata1)

            # Store track-2
            metadata2 = TrackMetadata(
                name="Ambient Waves",
                artists=["Chill Artist"],
                album="Relaxation",
                genres=["ambient", "electronic"],
            )
            await client.embed_text_and_store("track-2", metadata2)

            print(f"{GREEN}✓ Track embeddings stored{NC}")
        except Exception as e:
            print(f"{RED}✗ Failed to store track embeddings: {e}{NC}")
            return False
        print()

        # Test 5: Taste profile computation using SidecarClient method
        print(f"{YELLOW}[5/7] Testing taste profile computation...{NC}")
        try:
            import time
            now = int(time.time())
            # Use recent timestamps (within the last few days)
            interactions = [
                {
                    "track_id": "track-1",
                    "timestamp": now - (2 * 86400),  # 2 days ago
                    "signal_type": "full_play",
                    "seconds_played": 240.0,
                    "duration": 240.0,
                },
                {
                    "track_id": "track-2",
                    "timestamp": now - (1 * 86400),  # 1 day ago
                    "signal_type": "full_play",
                    "seconds_played": 180.0,
                    "duration": 180.0,
                },
                {
                    "track_id": "track-1",
                    "timestamp": now - (12 * 3600),  # 12 hours ago
                    "signal_type": "repeat",
                    "seconds_played": 240.0,
                    "duration": 240.0,
                },
            ]

            # Use the SidecarClient method (same as MA provider would use)
            user_id = "test-user-001"
            profile_response = await client.compute_taste_profile(
                user_id=user_id,
                interactions=interactions,
                cutoff_days=21,
            )

            if "profiles" in profile_response:
                profiles = profile_response["profiles"]
                print(f"{GREEN}✓ Taste profile computed successfully{NC}")
                print(f"Profiles created: {len(profiles)}")
            else:
                print(f"{RED}✗ Taste profile computation failed{NC}")
                print(f"Response: {profile_response}")
                return False
        except Exception as e:
            print(f"{RED}✗ Taste profile computation failed: {e}{NC}")
            return False
        print()

        # Test 6: Get recommendations using SidecarClient method
        print(f"{YELLOW}[6/7] Testing taste-based recommendations...{NC}")
        try:
            recommendations = await client.get_taste_recommendations(
                user_id=user_id,
                limit=10,
            )

            if "tracks" in recommendations:
                tracks = recommendations["tracks"]
                print(f"{GREEN}✓ Recommendations retrieved successfully{NC}")
                print(f"Recommendations returned: {len(tracks)}")
                if len(tracks) > 0:
                    print(f"  Top recommendation: {tracks[0].get('track_id', 'unknown')}")
            else:
                print(f"{RED}✗ Recommendations failed{NC}")
                print(f"Response: {recommendations}")
                return False
        except Exception as e:
            print(f"{RED}✗ Recommendations failed: {e}{NC}")
            return False
        print()

        # Test 7: Final health check
        print(f"{YELLOW}[7/7] Verifying sidecar stability...{NC}")
        try:
            health = await client.health_check()
            if "status" in health:
                print(f"{GREEN}✓ Sidecar remained stable after operations{NC}")
            else:
                print(f"{RED}✗ Sidecar health check failed{NC}")
                return False
        except Exception as e:
            print(f"{RED}✗ Sidecar health check failed: {e}{NC}")
            return False
        print()

        print("=" * 60)
        print(f"{GREEN}All Phase 1 Integration Tests Passed! ✓{NC}")
        print("=" * 60)
        return True

    finally:
        # Clean up
        await client.close()
        print(f"\n{YELLOW}Stopping sidecar...{NC}")
        sidecar.terminate()
        try:
            sidecar.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sidecar.kill()
        print("Sidecar stopped")


async def run_all_tests():
    """Run tests with both file and Qdrant storage backends."""
    print("=" * 60)
    print("Phase 1 Integration Verification")
    print("Testing both storage backends")
    print("=" * 60)

    results = {}

    # Test 1: File-based storage (usearch)
    print(f"\n{BLUE}[Storage Backend 1/2] File-based storage (usearch){NC}")
    try:
        results["file"] = await run_tests_with_storage(
            storage_mode="file",
            storage_args=["--storage-mode", "file"]
        )
    except Exception as e:
        print(f"{RED}File storage test failed with exception: {e}{NC}")
        results["file"] = False

    # Test 2: Qdrant storage
    print(f"\n{BLUE}[Storage Backend 2/2] Qdrant storage{NC}")
    try:
        results["qdrant"] = await run_tests_with_storage(
            storage_mode="qdrant",
            storage_args=["--storage-mode", "qdrant", "--qdrant-url", QDRANT_URL]
        )
    except Exception as e:
        print(f"{RED}Qdrant storage test failed with exception: {e}{NC}")
        results["qdrant"] = False

    # Print summary
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"File storage (usearch): {GREEN + '✓ PASS' + NC if results['file'] else RED + '✗ FAIL' + NC}")
    print(f"Qdrant storage:         {GREEN + '✓ PASS' + NC if results['qdrant'] else RED + '✗ FAIL' + NC}")
    print("=" * 60)

    all_passed = all(results.values())
    if all_passed:
        print(f"{GREEN}All storage backends passed!{NC}")
    else:
        print(f"{RED}Some storage backends failed!{NC}")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
