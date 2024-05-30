/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 4.0.1
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.microsoft.ml.lightgbm;

public class floatChunkedArray {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected floatChunkedArray(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(floatChunkedArray obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  @SuppressWarnings("deprecation")
  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        lightgbmlibJNI.delete_floatChunkedArray(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public floatChunkedArray(long chunk_size) {
    this(lightgbmlibJNI.new_floatChunkedArray(chunk_size), true);
  }

  public void add(float value) {
    lightgbmlibJNI.floatChunkedArray_add(swigCPtr, this, value);
  }

  public long get_add_count() {
    return lightgbmlibJNI.floatChunkedArray_get_add_count(swigCPtr, this);
  }

  public long get_chunks_count() {
    return lightgbmlibJNI.floatChunkedArray_get_chunks_count(swigCPtr, this);
  }

  public long get_last_chunk_add_count() {
    return lightgbmlibJNI.floatChunkedArray_get_last_chunk_add_count(swigCPtr, this);
  }

  public long get_chunk_size() {
    return lightgbmlibJNI.floatChunkedArray_get_chunk_size(swigCPtr, this);
  }

  public SWIGTYPE_p_p_float data() {
    long cPtr = lightgbmlibJNI.floatChunkedArray_data(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_p_float(cPtr, false);
  }

  public SWIGTYPE_p_p_void data_as_void() {
    long cPtr = lightgbmlibJNI.floatChunkedArray_data_as_void(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_p_void(cPtr, false);
  }

  public void coalesce_to(SWIGTYPE_p_float other, boolean all_valid_addresses) {
    lightgbmlibJNI.floatChunkedArray_coalesce_to__SWIG_0(swigCPtr, this, SWIGTYPE_p_float.getCPtr(other), all_valid_addresses);
  }

  public void coalesce_to(SWIGTYPE_p_float other) {
    lightgbmlibJNI.floatChunkedArray_coalesce_to__SWIG_1(swigCPtr, this, SWIGTYPE_p_float.getCPtr(other));
  }

  public float getitem(long chunk_index, long index_within_chunk, float on_fail_value) {
    return lightgbmlibJNI.floatChunkedArray_getitem(swigCPtr, this, chunk_index, index_within_chunk, on_fail_value);
  }

  public int setitem(long chunk_index, long index_within_chunk, float value) {
    return lightgbmlibJNI.floatChunkedArray_setitem(swigCPtr, this, chunk_index, index_within_chunk, value);
  }

  public void clear() {
    lightgbmlibJNI.floatChunkedArray_clear(swigCPtr, this);
  }

  public void release() {
    lightgbmlibJNI.floatChunkedArray_release(swigCPtr, this);
  }

  public boolean within_bounds(long chunk_index, long index_within_chunk) {
    return lightgbmlibJNI.floatChunkedArray_within_bounds(swigCPtr, this, chunk_index, index_within_chunk);
  }

  public void new_chunk() {
    lightgbmlibJNI.floatChunkedArray_new_chunk(swigCPtr, this);
  }

}
