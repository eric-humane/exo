# Exo Test Suite

This directory contains tests for the exo codebase, focusing particularly on the async components.

## Async Improvements Tests

The `test_async_improvements.py` file contains tests for the async callback system and other async-related improvements. These tests verify:

1. **AsyncCallback Class**: Tests both sync and async callback functionality
2. **AsyncCallbackSystem**: Tests the callback registration and triggering system
3. **Wait Functionality**: Tests the ability to wait for conditions using both sync and async predicates
4. **Node Class**: Tests async callbacks in the Node orchestration class
5. **TaskManager**: Tests the concept for a centralized task management system (for future implementation)

## Running Tests

From the project root:

```bash
# Run all async tests
./run_async_tests.sh

# Or run specific test module
python -m exo.tests.test_async_improvements
```

## Writing New Tests

When writing new async tests:

1. Use `unittest.IsolatedAsyncioTestCase` for proper asyncio test isolation
2. Test both synchronous and asynchronous code paths
3. Properly clean up resources in test teardown
4. Use appropriate timeouts to avoid hanging tests
5. Add descriptive docstrings to test methods

## Future Improvements

Planned enhancements to the test suite:

1. Comprehensive coverage of all async components
2. Performance testing for async operations
3. Stress testing for concurrent operations
4. Mocks for external dependencies to enable isolated testing